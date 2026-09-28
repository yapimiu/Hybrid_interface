# Hybrid Interface

Приложение принимает первый найденный LSL поток типа `EEG` или `Signal` и показывает сигналы каналов в окне Qt. При обрыве соединения поиск запускается снова.

## Установка и запуск

```bash
git clone https://github.com/yapimiu/Hybrid_interface.git
cd Hybrid_interface
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.main
```

На Windows активация окружения: `.venv\Scripts\activate`.

## Массив данных

Последний принятый блок доступен в `window.signal_array` (`numpy.ndarray`). Его форма — `(число отсчётов, число каналов)`. Это сырые значения LSL; новый блок заменяет предыдущий. До первого блока массив пустой. Переменная принадлежит экземпляру `SignalViewerWindow` в `app/viewer.py` и обновляется в главном потоке Qt.

Графики сохраняют только данные текущего окна отображения. Все каналы показываются автоматически; масштаб по X и Y и показатели под графиками доступны в интерфейсе. Окно можно прокручивать вверх и вниз, в том числе колесом мыши поверх графиков.

Под графиками отображается полный последний принятый массив с его размером. Значения можно выделить и скопировать. Поле обновляется каждые 250 мс и показывает сырые значения всех каналов.

Встроенного генератора или тестового LSL потока нет. Приложение принимает данные внешнего источника.

## Сборка

```bash
pip install -r requirements-dev.txt
python scripts/build_app.py
```

На macOS результат — `dist/Hybrid Interface.app`. На Windows — `dist/HybridInterface.exe`.

Для автоматической установки зависимостей и сборки на Windows запустите `build_windows.bat`. Подробности — в `BUILD_WINDOWS.md`.

В `build_templates/github-actions.yml` подготовлен шаблон сборки приложений для Windows и macOS. Чтобы включить GitHub Actions, добавьте его в репозиторий как `.github/workflows/build.yml` через GitHub или с токеном, имеющим право `workflow`. После успешной сборки готовые архивы появятся в разделе Actions → Build applications → Artifacts.

## Структура

- `app/viewer.py` — приём LSL, графики и вывод массива.
- `app/main.py` — точка входа.
- `scripts/build_app.py` — сборка приложения.
- `hybrid_interface.spec` — настройки PyInstaller.

Лицензия MIT; обязательное уведомление об авторстве сохранено в `LICENSE`.
