import io
import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from tensorflow.keras.models import model_from_json

app = Flask(__name__)

# ✅ CORS: Only allow requests from your GitHub Pages
CORS(app, resources={r"/solve": {"origins": "https://satyaprakashmohanty13.github.io"}})

# ✅ Extra: Block other referers too
@app.before_request
def block_unauthorized_referer():
    allowed_referer = "https://satyaprakashmohanty13.github.io"
    referer = request.headers.get("Referer", "")
    if referer and not referer.startswith(allowed_referer):
        return jsonify({"error": "Unauthorized referer"}), 403

# Load model
with open('model/model.json') as f:
    model = model_from_json(f.read())
model.load_weights('model/model.h5')
model.make_predict_function()

def preprocess_cell(cell):
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized = cv2.resize(thresh, (28, 28))
    normed = resized.astype('float32') / 255.0
    return normed.reshape(1, 28, 28, 1)

def extract_grid(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 3)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if len(approx) != 4:
        raise ValueError("Could not find Sudoku grid.")

    pts = approx.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)]
    ], dtype='float32')

    side = max([
        np.linalg.norm(rect[0] - rect[1]),
        np.linalg.norm(rect[1] - rect[2]),
        np.linalg.norm(rect[2] - rect[3]),
        np.linalg.norm(rect[3] - rect[0])
    ])

    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (int(side), int(side)))
    return warp, M, int(side)

def split_cells(grid_img):
    side = grid_img.shape[0]
    cell_size = side // 9
    cells = []
    for i in range(9):
        row = []
        for j in range(9):
            cell = grid_img[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            row.append(cell)
        cells.append(row)
    return cells

def read_puzzle(warp):
    cells = split_cells(warp)
    puzzle = np.zeros((9, 9), dtype=int)
    for i in range(9):
        for j in range(9):
            cell = cells[i][j]
            if cv2.countNonZero(cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)) > (cell.shape[0] * cell.shape[1] * 0.03):
                digit = model.predict(preprocess_cell(cell)).argmax()
                puzzle[i, j] = digit
    return puzzle

def solve(puzzle):
    def find_empty(p):
        for x in range(9):
            for y in range(9):
                if p[x, y] == 0:
                    return x, y
        return None

    def valid(p, r, c, val):
        if val in p[r, :] or val in p[:, c]:
            return False
        br, bc = 3 * (r // 3), 3 * (c // 3)
        if val in p[br:br+3, bc:bc+3]:
            return False
        return True

    pos = find_empty(puzzle)
    if pos is None:
        return True
    r, c = pos
    for n in range(1, 10):
        if valid(puzzle, r, c, n):
            puzzle[r, c] = n
            if solve(puzzle):
                return True
            puzzle[r, c] = 0
    return False

def overlay_solution(orig, warp, M_inv, diff):
    side = warp.shape[0]
    cell_size = side // 9
    for i in range(9):
        for j in range(9):
            if diff[i, j] != 0:
                x = j * cell_size + cell_size // 3
                y = (i + 1) * cell_size - cell_size // 6
                cv2.putText(warp, str(diff[i, j]), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return cv2.warpPerspective(warp, M_inv, (orig.shape[1], orig.shape[0]),
                               flags=cv2.WARP_INVERSE_MAP)

@app.route('/solve', methods=['POST'])
def solve_endpoint():
    file = request.files.get('sudoku')
    if not file:
        return jsonify(error='No file uploaded'), 400

    img = Image.open(file.stream).convert('RGB')
    open_cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    try:
        warp, M, side = extract_grid(open_cv_img)
        puzzle = read_puzzle(warp).copy()
        solved = puzzle.copy()
        if not solve(solved):
            return jsonify(error='Unsolvable puzzle'), 400

        M_inv = np.linalg.inv(M)
        diff = solved - puzzle
        out = overlay_solution(open_cv_img, warp, M_inv, diff)

        _, buf = cv2.imencode('.jpg', out)
        return send_file(io.BytesIO(buf.tobytes()),
                         mimetype='image/jpeg')
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
