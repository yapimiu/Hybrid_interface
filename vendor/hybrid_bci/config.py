# Окно (в полноэкранном режиме — базовая логика координат; см. PANEL_X_FRACTION)
WINDOW_SIZE = (1200, 800)
WINDOW_COLOR = "black"
GRID_SIZE = 3

# Визуал плиток (логика «клетки»; масштаб отдельно)
TILE_SIZE_PX = 120
TILE_SPACING_PX = 20
TILE_DISPLAY_SCALE = 2.5
TILE_DEFAULT_COLOR = "#333333"
TILE_LINE_COLOR = "#555555"

# Пауза между полными кругами (перестановками 9 плиток), сек
DEFAULT_INTER_BLOCK_S = 0.8
INTER_BLOCK_MIN, INTER_BLOCK_MAX = 0.0, 5.0

# ISI — пауза после off до следующей вспышки, сек
DEFAULT_ISI = 0.10
ISI_MIN, ISI_MAX = 0.01, 2.0

# Длительность «горения» одной плитки, сек
DEFAULT_FLASH_DURATION = 0.10
FLASH_MIN, FLASH_MAX = 0.01, 0.5

# Сколько раундов (1 раунд = одна перестановка всех 9 плиток)
DEFAULT_SEQUENCES = 12
SEQUENCES_MIN, SEQUENCES_MAX = 1, 40

# Cue: подсветка целевой плитки, сек
DEFAULT_CUE_S = 2.0
CUE_MIN, CUE_MAX = 0.3, 10.0

# Ready: крест фиксации перед стимом, сек
DEFAULT_READY_S = 1.5
READY_MIN, READY_MAX = 0.2, 10.0

# Стартовая целевая плитка (0–8)
DEFAULT_TARGET_ID = 0

# Кнопки (от центра, пиксели; для 1200×800)
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 50
START_BUTTON_POS = (-100, -280)
STOP_BUTTON_POS = (100, -280)

# Панель настроек: X = доля * (половина ширины окна). Меньше — левее, ближе к сетке;
# при 0.72+ на широких/масштабированных экранах панель уезжала за правый край.
PANEL_X_FRACTION = 0.40
# Первая строка панели (y вверх от центра), шаг вниз (y уменьшается)
PANEL_FIRST_ROW_Y = 260
PANEL_ROW_DY = 36
# Поле ввода
PANEL_TB_W = 72
PANEL_TB_H = 26
PANEL_LETTER_H = 11
# Смещение подписи влево от центра полей (уменьшите |...|, чтобы колонка была уже)
PANEL_LABEL_OFFSET = -95

# Подсказка оператору (мелким, внизу)
OPERATOR_HINT_Y = -320
OPERATOR_HINT_H = 12

# Устаревшие имена (совместимость, если что-то импортирует)
ISI_RANGE = (ISI_MIN, ISI_MAX)
FLASH_DURATION_RANGE = (FLASH_MIN, FLASH_MAX)
SLIDER_GRANULARITY = 0.01
SEQUENCES_RANGE = list(range(SEQUENCES_MIN, SEQUENCES_MAX + 1))
FLASH_COLOR = "white"
FIXATION_CROSS_COLOR = "white"
FIXATION_CROSS_SIZE = 20
# Алиасы для кода, ожидающего CUE_DURATION / READY_DURATION
CUE_DURATION = DEFAULT_CUE_S
READY_DURATION = DEFAULT_READY_S
CUE_COLOR = "blue"
STIM_COLOR = "white"
