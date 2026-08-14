from __future__ import annotations


APP_STYLESHEET = """
QMainWindow {
    background: #f6f0e5;
}
QWidget {
    color: #24323a;
    font-family: "Segoe UI", "Trebuchet MS", sans-serif;
    font-size: 10pt;
}
QFrame#SidebarCard, QFrame#HeaderCard, QFrame#StatsCard {
    background: #fffaf2;
    border: 1px solid #d8cdbf;
    border-radius: 16px;
}
QScrollArea#SidebarScroll {
    background: transparent;
    border: none;
}
QScrollArea#SidebarScroll > QWidget > QWidget {
    background: transparent;
}
QFrame#CustomEditor {
    background: #fbf3e8;
    border: 1px solid #d8cdbf;
    border-radius: 12px;
}
QLabel#TitleLabel {
    font-size: 20pt;
    font-weight: 700;
    color: #16323d;
}
QLabel#SubtitleLabel {
    font-size: 10pt;
    color: #5b6d73;
}
QLabel#SectionLabel {
    font-size: 11pt;
    font-weight: 700;
    color: #184a57;
}
QLabel#HintLabel {
    font-size: 8pt;
    color: #7a6b59;
}
QLabel#HeaderFieldLabel {
    font-size: 8pt;
    font-weight: 700;
    color: #5d6e73;
}
QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {
    background: #fffdf8;
    color: #24323a;
    border: 1px solid #ccbfae;
    border-radius: 10px;
    padding: 6px 10px;
}
QComboBox#HeaderCombo {
    min-width: 150px;
}
QComboBox::drop-down {
    border: none;
    width: 28px;
}
QComboBox QAbstractItemView {
    background: #fffdf8;
    color: #24323a;
    border: 1px solid #ccbfae;
    selection-background-color: #dfeee9;
    selection-color: #16323d;
    padding: 4px;
    outline: 0;
}
QPushButton {
    background: #1f7a8c;
    color: white;
    border: none;
    border-radius: 11px;
    padding: 8px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background: #246d7b;
}
QPushButton#CompactButton {
    padding: 6px 8px;
    min-height: 26px;
}
QPushButton#SecondaryButton {
    background: #e8dccb;
    color: #25414a;
}
QPushButton#SecondaryButton:hover {
    background: #d9ccb7;
}
QPushButton#CompactSecondaryButton {
    background: #e8dccb;
    color: #25414a;
    padding: 6px 8px;
    min-height: 26px;
}
QPushButton#CompactSecondaryButton:hover {
    background: #d9ccb7;
}
QPushButton#ClearButton {
    background: #f2d3cc;
    color: #7b2d25;
    padding: 6px 8px;
    min-height: 26px;
}
QPushButton#ClearButton:hover {
    background: #e7bdb4;
}
QScrollBar:vertical {
    background: #f4eadc;
    width: 10px;
    margin: 2px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #c8b9a5;
    border-radius: 5px;
    min-height: 28px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QSlider::groove:horizontal {
    border: none;
    height: 8px;
    background: #dccfbc;
    border-radius: 4px;
}
QSlider::handle:horizontal {
    background: #d1792b;
    width: 18px;
    margin: -5px 0;
    border-radius: 9px;
}
QTextEdit {
    background: #fffdf8;
    border: 1px solid #d7c9b7;
    border-radius: 10px;
    padding: 8px;
}
QTableWidget {
    background: #fffdf8;
    alternate-background-color: #f7efe3;
    color: #24323a;
    border: 1px solid #d7c9b7;
    border-radius: 10px;
    gridline-color: #e4d8c7;
}
QHeaderView::section {
    background: #eadfce;
    color: #244650;
    border: none;
    padding: 5px;
    font-weight: 700;
}
QTableWidget::item:selected {
    background: #dfeee9;
    color: #16323d;
}
"""
