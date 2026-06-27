import re

import numpy as np

from PyQt5.QtWidgets import QDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class TransformInputDialog(QDialog):
    """Reusable 4x4 matrix input dialog."""

    def __init__(self, parent=None, title: str = "Input 4x4 Matrix", prompt_text: str = None,
                 matrix: np.ndarray = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(600)

        default_prompt = (
            "Enter the 4x4 matrix:\n"
            "(You can paste the entire terminal log block directly into the top-left box)"
        )

        layout = QVBoxLayout(self)
        self.prompt_label = QLabel(prompt_text or default_prompt)
        self.prompt_label.setWordWrap(True)
        layout.addWidget(self.prompt_label)

        grid_layout = QGridLayout()
        self.inputs = []

        for row in range(4):
            row_inputs = []
            for col in range(4):
                line_edit = QLineEdit()
                line_edit.setText("1.0" if row == col else "0.0")
                grid_layout.addWidget(line_edit, row, col)
                row_inputs.append(line_edit)
            self.inputs.append(row_inputs)

        self.inputs[0][0].textChanged.connect(self._try_smart_paste)
        layout.addLayout(grid_layout)

        button_layout = QHBoxLayout()
        reset_btn = QPushButton("Reset to Identity")
        reset_btn.setToolTip("Reset the matrix to identity (no transformation).")
        ok_btn = QPushButton("Apply")
        ok_btn.setToolTip("Apply the entered 4x4 transformation matrix.")
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setToolTip("Close this dialog without applying the matrix.")

        reset_btn.clicked.connect(self._reset_to_identity)
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(reset_btn)
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(ok_btn)
        layout.addLayout(button_layout)

        if matrix is not None:
            self.set_matrix(matrix)

    def set_prompt_text(self, text: str):
        self.prompt_label.setText(text)

    def set_matrix(self, matrix: np.ndarray):
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("Matrix must be 4x4")

        for row in range(4):
            for col in range(4):
                widget = self.inputs[row][col]
                widget.blockSignals(True)
                widget.setText(str(matrix[row, col]))
                widget.blockSignals(False)

    def _try_smart_paste(self, text):
        """Detect matrix paste text and auto-fill the dialog."""
        if '[' not in text or ']' not in text:
            return

        floats = []
        blocks = re.findall(r'\[(.*?)\]', text)
        for block in blocks:
            for val in block.split(','):
                val = val.strip()
                if val:
                    try:
                        floats.append(float(val))
                    except ValueError:
                        pass

        if len(floats) == 16:
            self.inputs[0][0].textChanged.disconnect(self._try_smart_paste)
            for row in range(4):
                for col in range(4):
                    self.inputs[row][col].setText(str(floats[row * 4 + col]))
            self.inputs[0][0].textChanged.connect(self._try_smart_paste)

    def get_matrix(self):
        matrix = np.eye(4, dtype=np.float64)
        for row in range(4):
            for col in range(4):
                try:
                    value_text = self.inputs[row][col].text().strip('[], ')
                    matrix[row, col] = float(value_text)
                except ValueError:
                    print(f"Failed to parse value at {row},{col}. Defaulting to 0.0")
                    matrix[row, col] = 0.0
        return matrix

    def _reset_to_identity(self):
        for row in range(4):
            for col in range(4):
                self.inputs[row][col].setText("1.0" if row == col else "0.0")
