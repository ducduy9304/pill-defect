# Look of the live app: the dark theme of DemoTemplate_thadosoft/PyUICBasicDemo.py (the parts
# this app uses), plus the verdict.

OK_COLOR = "#3fb950"
NG_COLOR = "#f85149"
IDLE_COLOR = "#8b949e"

APP_STYLESHEET = """
QMainWindow, QWidget#centralwidget {
    background-color: #1c2128;
    color: #e6edf3;
    font-family: "Segoe UI", "Ubuntu", sans-serif;
    font-size: 16px;
}

QLabel#titleLabel {
    color: #f0f3f6;
    font-size: 24px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

QLabel#subtitleLabel {
    color: #8b949e;
    font-size: 15px;
}

QLabel#viewCaption {
    color: #8b949e;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.4px;
}

QLabel#raw,
QLabel#result {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    color: #484f58;
    font-size: 18px;
}

QFrame#sidePanel {
    background-color: #22272e;
    border: 1px solid #30363d;
    border-radius: 10px;
}

QGroupBox {
    font-size: 15px;
    background-color: transparent;
    border: 1px solid #30363d;
    border-radius: 8px;
    margin-top: 12px;
    padding: 14px 10px 10px 10px;
    font-weight: 600;
    color: #c9d1d9;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #8b949e;
    font-size: 14px;
    font-weight: 600;
}

QComboBox {
    font-size: 17px;
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 10px;
    color: #e6edf3;
    min-height: 28px;
}

QComboBox:hover {
    border-color: #58a6ff;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background-color: #161b22;
    border: 1px solid #30363d;
    selection-background-color: #1f6feb;
    color: #e6edf3;
    outline: none;
}

QPushButton {
    font-size: 15px;
    background-color: #30363d;
    border: 1px solid #484f58;
    border-radius: 6px;
    padding: 8px 12px;
    color: #e6edf3;
    font-weight: 500;
    min-height: 28px;
}

QPushButton:hover {
    background-color: #3d444d;
    border-color: #6e7681;
}

QPushButton:pressed {
    background-color: #21262d;
}

QPushButton:disabled {
    background-color: #21262d;
    border-color: #30363d;
    color: #484f58;
}

QPushButton#checkbtn {
    background-color: #238636;
    border-color: #238636;
    color: #ffffff;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 1px;
    min-height: 40px;
}

QPushButton#checkbtn:hover {
    background-color: #2ea043;
    border-color: #2ea043;
}

QRadioButton {
    color: #c9d1d9;
    spacing: 8px;
}

QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border-radius: 9px;
    border: 2px solid #484f58;
    background: #0d1117;
}

QRadioButton::indicator:checked {
    border-color: #58a6ff;
    background: qradialgradient(
        cx:0.5, cy:0.5, radius:0.4,
        fx:0.5, fy:0.5,
        stop:0 #58a6ff, stop:1 #0d1117
    );
}

QLabel#procTime {
    color: #3fb950;
    font-size: 18px;
    font-weight: 600;
    font-family: "Cascadia Mono", "DejaVu Sans Mono", monospace;
}

QLabel#verdictLabel {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: 2px;
}

QLabel#logoLabel {
    background: transparent;
    border: none;
}

QPushButton#stepRow {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0;
    text-align: left;
    min-height: 52px;
}

QPushButton#stepRow:hover {
    border-color: #6e7681;
}

QPushButton#stepRow:checked {
    background-color: #132339;
    border: 2px solid #58a6ff;
}

QLabel#stepName {
    color: #e6edf3;
    font-size: 17px;
    font-weight: 600;
    background: transparent;
}


QLabel#stepStatus {
    font-size: 20px;
    font-weight: 800;
    background: transparent;
}

QLabel#faultsLabel {
    color: #f0b8b4;
    font-size: 15px;
}

QLabel#saveLabel {
    color: #8b949e;
    font-size: 14px;
}

QDoubleSpinBox {
    font-size: 16px;
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 4px 8px;
    color: #e6edf3;
    min-height: 24px;
}

QFormLayout QLabel, QGroupBox QLabel {
    color: #c9d1d9;
    font-size: 15px;
}
"""
