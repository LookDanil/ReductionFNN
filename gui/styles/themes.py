"""Стили тем оформления"""

LIGHT_QSS = """
QMainWindow { background-color: #f5f5f5; }
QWidget { background-color: #f5f5f5; color: #222222; }
QLabel { color: #222222; background: transparent; }
QGroupBox { font-weight: bold; border: 2px solid #cccccc; border-radius: 5px; margin-top: 10px; padding-top: 15px; color: #222222; background-color: #ffffff; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #1565C0; background-color: #ffffff; }
QPushButton { background-color: #e0e0e0; color: #222222; border: 1px solid #bdbdbd; border-radius: 4px; padding: 6px 12px; }
QPushButton:hover { background-color: #d0d0d0; }
QPushButton#startButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px 20px; border-radius: 5px; font-size: 14px; border: none; }
QPushButton#startButton:disabled { background-color: #bdbdbd; color: #757575; }
QPushButton#startButton:hover:!disabled { background-color: #43A047; }
QPushButton#stopButton { background-color: #f44336; color: white; font-weight: bold; padding: 10px 20px; border-radius: 5px; font-size: 14px; border: none; }
QPushButton#stopButton:disabled { background-color: #bdbdbd; color: #757575; }
QPushButton#stopButton:hover:!disabled { background-color: #d32f2f; }
QTabWidget::pane { border-top: 3px solid #1565C0; background-color: #ffffff; }
QTabBar::tab { background: #e8e8e8; padding: 10px 20px; margin-right: 2px; font-size: 13px; color: #333333; border: 1px solid #cccccc; border-bottom: none; }
QTabBar::tab:selected { background: #1565C0; color: white; border-color: #1565C0; }
QTabBar::tab:disabled { background: #f5f5f5; color: #999999; }
QSpinBox { background: white; color: #222222; border: 1px solid #bdbdbd; padding: 4px; border-radius: 3px; }
QComboBox { background: white; color: #222222; border: 1px solid #bdbdbd; padding: 4px 8px; border-radius: 3px; }
QComboBox QAbstractItemView { background: white; color: #222222; selection-background-color: #1565C0; selection-color: white; border: 1px solid #bdbdbd; }
QTableWidget { background: white; color: #222222; gridline-color: #e0e0e0; border: 1px solid #bdbdbd; }
QHeaderView::section { background: #e8e8e8; color: #222222; padding: 4px; border: 1px solid #cccccc; font-weight: bold; }
QTextEdit { background: white; color: #222222; border: 1px solid #bdbdbd; }
QDialog { background: #f5f5f5; }
QMessageBox { background: #ffffff; }
QMessageBox QLabel { color: #222222; background: transparent; }
QStatusBar { background: #e8e8e8; color: #333333; }
QMenuBar { background: #e8e8e8; color: #333333; }
QMenuBar::item:selected { background: #1565C0; color: white; }
QMenu { background: white; color: #222222; border: 1px solid #cccccc; }
QMenu::item:selected { background: #1565C0; color: white; }
QSplitter::handle { background: #cccccc; }
"""

DARK_QSS = """
QMainWindow { background-color: #1a1a2e; }
QWidget { background-color: #1a1a2e; color: #e0e0e0; }
QLabel { color: #e0e0e0; background: transparent; }
QGroupBox { font-weight: bold; border: 2px solid #444466; border-radius: 5px; margin-top: 10px; padding-top: 15px; color: #e0e0e0; background-color: #222240; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #64b5f6; background-color: #222240; }
QPushButton { background-color: #333355; color: #e0e0e0; border: 1px solid #555577; border-radius: 4px; padding: 6px 12px; }
QPushButton:hover { background-color: #444466; }
QPushButton#startButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px 20px; border-radius: 5px; font-size: 14px; border: none; }
QPushButton#startButton:disabled { background-color: #444466; color: #777799; }
QPushButton#startButton:hover:!disabled { background-color: #43A047; }
QPushButton#stopButton { background-color: #f44336; color: white; font-weight: bold; padding: 10px 20px; border-radius: 5px; font-size: 14px; border: none; }
QPushButton#stopButton:disabled { background-color: #444466; color: #777799; }
QPushButton#stopButton:hover:!disabled { background-color: #d32f2f; }
QTabWidget::pane { border-top: 3px solid #64b5f6; background-color: #222240; }
QTabBar::tab { background: #2a2a44; padding: 10px 20px; margin-right: 2px; font-size: 13px; color: #bbbbcc; border: 1px solid #444466; border-bottom: none; }
QTabBar::tab:selected { background: #64b5f6; color: #111122; border-color: #64b5f6; }
QTabBar::tab:disabled { background: #1a1a2e; color: #555566; }
QSpinBox { background: #2a2a44; color: #e0e0e0; border: 1px solid #555577; padding: 4px; border-radius: 3px; }
QComboBox { background: #2a2a44; color: #e0e0e0; border: 1px solid #555577; padding: 4px 8px; border-radius: 3px; }
QComboBox QAbstractItemView { background: #2a2a44; color: #e0e0e0; selection-background-color: #64b5f6; selection-color: #111122; border: 1px solid #555577; }
QTableWidget { background: #2a2a44; color: #e0e0e0; gridline-color: #444466; border: 1px solid #555577; }
QHeaderView::section { background: #333355; color: #e0e0e0; padding: 4px; border: 1px solid #555577; font-weight: bold; }
QTextEdit { background: #2a2a44; color: #e0e0e0; border: 1px solid #555577; }
QDialog { background: #1a1a2e; }
QMessageBox { background: #222240; }
QMessageBox QLabel { color: #e0e0e0; background: transparent; }
QStatusBar { background: #16213e; color: #bbbbcc; }
QMenuBar { background: #16213e; color: #bbbbcc; }
QMenuBar::item:selected { background: #64b5f6; color: #111122; }
QMenu { background: #2a2a44; color: #e0e0e0; border: 1px solid #555577; }
QMenu::item:selected { background: #64b5f6; color: #111122; }
QSplitter::handle { background: #444466; }
"""