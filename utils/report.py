import sys
import bisect
import numpy as np
import pandas as pd
import pyqtgraph as pg
from collections import deque, defaultdict
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QGridLayout,
    QLabel,
    QSpacerItem,
    QSizePolicy,
    QVBoxLayout,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFrame,
    QGraphicsView,
    QCheckBox,
    QHBoxLayout,
    QPushButton,
    QMessageBox,
    QMenu,
    QAction,
    QLineEdit,
    QWidgetAction,
    QTextEdit,
)
from PyQt5.QtGui import QFont, QColor, QPainter, QPicture, QBrush, QPen, QPolygonF
from PyQt5.QtCore import Qt, QPointF, QRectF
from typing import Callable

from core.models import BacktestingResult


class FilterButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._active = False
        self.setFixedSize(16, 16)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("QPushButton { border: none; background: transparent; }")

    def set_active(self, active):
        self._active = active
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        color = QColor("#d14545") if self._active else QColor("#999999")
        if self._active:
            p.setBrush(QBrush(color))
        else:
            p.setBrush(QBrush(color))
        p.setPen(Qt.NoPen)
        w, h = self.width(), self.height()
        funnel = [
            QPointF(w * 0.1, h * 0.1),
            QPointF(w * 0.9, h * 0.1),
            QPointF(w * 0.55, h * 0.55),
            QPointF(w * 0.55, h * 0.9),
            QPointF(w * 0.45, h * 0.9),
            QPointF(w * 0.45, h * 0.55),
        ]
        p.drawPolygon(QPolygonF(funnel))
        p.end()


class ClickableLegend(pg.LegendItem):
    """支持点击隐藏/显示曲线的图例"""

    def __init__(self, size=None, offset=None, **kwargs):
        super().__init__(size=size, offset=offset, **kwargs)
        self._curve_items = {}

    def addItem(self, item, name):
        super().addItem(item, name)
        label = self.items[-1][1]
        self._curve_items[label] = item
        label.setAcceptHoverEvents(True)
        label.setCursor(Qt.PointingHandCursor)
        label.setToolTip("点击隐藏/显示")
        label.mousePressEvent = lambda ev, lbl=label: self._toggle_curve(lbl)
        label.hoverEvent = lambda ev, lbl=label: self._on_hover(ev, lbl)

    def _toggle_curve(self, label):
        curve = self._curve_items.get(label)
        if curve is not None:
            is_visible = curve.isVisible()
            curve.setVisible(not is_visible)
            label.setText(
                f"<span style='color: #999999; text-decoration: line-through;'>{label.text}</span>"
                if is_visible else label.text
            )

    def _on_hover(self, ev, label):
        if ev.isEnter():
            label.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        elif ev.isExit():
            label.setFont(QFont("Microsoft YaHei", 10, QFont.Normal))


class CandlestickItem(pg.GraphicsObject):
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data

        if data:
            self.low_min = min(d["low"] for d in data)
            self.high_max = max(d["high"] for d in data)
            price_range = self.high_max - self.low_min
            self.y_min = self.low_min - price_range * 0.05
            self.y_max = self.high_max + price_range * 0.05
        else:
            self.low_min = 0
            self.high_max = 1
            self.y_min = 0
            self.y_max = 1

        self.generatePicture()

    def generatePicture(self):
        self.picture = QPicture()
        p = QPainter(self.picture)

        if not self.data:
            p.end()
            return

        bar_width = 0.6
        line_half_width = bar_width / 2
        line_thickness = 1.5

        for d in self.data:
            t = d["time"]
            open_price = d["open"]
            high = d["high"]
            low = d["low"]
            close = d["close"]

            if close > open_price:
                linestyle = "阳线"
                p.setBrush(pg.mkBrush("#d14545"))
                p.setPen(pg.mkPen("#d14545", width=line_thickness))
            elif close < open_price:
                linestyle = "阴线"
                p.setBrush(pg.mkBrush("#3f993f"))
                p.setPen(pg.mkPen("#3f993f", width=line_thickness))
            else:
                linestyle = "平盘"
                p.setPen(pg.mkPen("#808080", width=line_thickness))
                p.drawLine(
                    QPointF(t - line_half_width, open_price),
                    QPointF(t + line_half_width, open_price),
                )

            if abs(high - low) > 1e-6:
                if linestyle == "阳线":
                    color = "#d14545"
                elif linestyle == "阴线":
                    color = "#3f993f"
                elif linestyle == "平盘":
                    color = "#808080"
                p.setPen(pg.mkPen(color, width=line_thickness))
                p.drawLine(QPointF(t, low), QPointF(t, high))

            if linestyle != "平盘":
                rect_top = min(open_price, close)
                rect_height = abs(close - open_price)
                p.drawRect(QRectF(t - bar_width / 2, rect_top, bar_width, rect_height))

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        if not self.data:
            return QRectF(0, 0, 1, 1)

        times = [d["time"] for d in self.data]
        lows = [d["low"] for d in self.data]
        highs = [d["high"] for d in self.data]

        x_min = min(times) - 0.5
        x_max = max(times) + 0.5
        y_min = min(lows)
        y_max = max(highs)

        price_range = y_max - y_min
        y_min_padded = y_min - price_range * 0.05
        y_max_padded = y_max + price_range * 0.05

        return QRectF(x_min, y_min_padded, x_max - x_min, y_max_padded - y_min_padded)


class FullValueAxis(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enableAutoSIPrefix(False)

    def tickStrings(self, values, scale, spacing):
        return [f"{int(value):,}" for value in values]


class DateAxis(pg.AxisItem):
    def __init__(self, x_values, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_values = x_values
        self.setLabel(text="交易日", units=None, **{"color": "k", "font-size": "12pt"})
        self.enableAutoSIPrefix(False)
        self.seen_indices_in_draw = set()

    def tickStrings(self, values, scale, spacing):
        strings = []
        for v in values:
            v_int = int(round(v))
            if 0 <= v_int < len(self.x_values):
                if v_int in self.seen_indices_in_draw:
                    strings.append("")
                else:
                    strings.append(self.x_values[v_int])
                    self.seen_indices_in_draw.add(v_int)
            else:
                strings.append("")
        return strings

    def generateDrawSpecs(self, p):
        self.seen_indices_in_draw.clear()
        labelSpec, tickSpecs, textSpecs = super().generateDrawSpecs(p)

        if not textSpecs:
            return (labelSpec, tickSpecs, textSpecs)

        font_metrics = p.fontMetrics()
        char_width = font_metrics.width("0000-00-00")
        available_width = self.width()
        max_ticks = max(1, int(available_width / (char_width * 1.5)))

        if len(textSpecs) > max_ticks:
            step = max(1, len(textSpecs) // max_ticks)
            thinned_textSpecs = []
            for i in range(0, len(textSpecs), step):
                thinned_textSpecs.append(textSpecs[i])
            if not thinned_textSpecs or textSpecs[-1] != thinned_textSpecs[-1]:
                thinned_textSpecs.append(textSpecs[-1])
            return (labelSpec, tickSpecs, thinned_textSpecs)

        if len(textSpecs) >= 2:
            try:
                positions = [spec[0].x() for spec in textSpecs]
                min_spacing = min(
                    abs(positions[i + 1] - positions[i])
                    for i in range(len(positions) - 1)
                )
                if min_spacing < char_width:
                    step = max(2, int(char_width / min_spacing))
                    thinned_textSpecs = textSpecs[::step]
                    if not thinned_textSpecs or textSpecs[-1] != thinned_textSpecs[-1]:
                        thinned_textSpecs.append(textSpecs[-1])
                    return (labelSpec, tickSpecs, thinned_textSpecs)
            except (AttributeError, IndexError):
                pass

        return (labelSpec, tickSpecs, textSpecs)


class BacktestReportWindow(QMainWindow):
    def __init__(self, result: "BacktestingResult", parent=None):
        super().__init__(parent)
        self.result = result
        self.setWindowTitle("回测报告")
        self.setGeometry(100, 100, 1858, 1082)
        self.result.prepare_data()
        self.plot_df = self.result.df
        if self.result.trade_start_date and self.result.df is not None and not self.result.df.empty:
            trade_start_dt = pd.to_datetime(self.result.trade_start_date)
            dt_series = pd.to_datetime(self.result.df["datetime"])
            self.plot_df = self.result.df[dt_series >= trade_start_dt].reset_index(drop=True)
        self.trade_markers: dict[int, dict] = {}
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self._user_adjusted_columns = set()
        self._column_widths = {}
        self._instrument_hidden = False
        self._original_instruments = []
        self._column_filters = {}
        self._filter_buttons = {}
        self._saved_sort_column = -1
        self._saved_sort_order = Qt.AscendingOrder

        self.overview_tab = self._create_overview_tab()
        self.tabs.addTab(self.overview_tab, "总览")

        if self.result.config.show_kline:
            self.kline_tab = self._create_kline_chart_tab()
            self.tabs.addTab(self.kline_tab, "K线图")

        self.daily_pnl_tab = self._create_daily_pnl_tab()
        self.tabs.addTab(self.daily_pnl_tab, "每日收益")

        self.trade_analysis_tab = self._create_trade_log_tab()
        self.tabs.addTab(self.trade_analysis_tab, "交易明细")

        self.benchmark_tab = self._create_benchmark_tab()
        self.tabs.addTab(self.benchmark_tab, "基准对比")

    def _create_crosshair_items(
        self, plot_widget: pg.PlotWidget
    ) -> tuple[pg.InfiniteLine, pg.InfiniteLine, QTextEdit]:
        vLine = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("k", style=Qt.DashLine)
        )
        hLine = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen("k", style=Qt.DashLine)
        )
        vLine.hide()
        hLine.hide()
        plot_widget.addItem(vLine, ignoreBounds=True)
        plot_widget.addItem(hLine, ignoreBounds=True)

        tooltip_edit = QTextEdit(plot_widget)
        tooltip_edit.setReadOnly(True)
        tooltip_edit.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        tooltip_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        tooltip_edit.setFocusPolicy(Qt.WheelFocus)
        tooltip_edit.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        tooltip_edit.setMinimumWidth(420)
        tooltip_edit.setStyleSheet(
            "QTextEdit {"
            "  background-color: rgba(255, 255, 255, 235);"
            "  border: 1px solid #cccccc;"
            "  padding: 6px;"
            "  font-size: 15px;"
            "  font-family: Microsoft YaHei;"
            "}"
            "QTextEdit QScrollBar:vertical {"
            "  width: 8px;"
            "  background: rgba(240, 240, 240, 200);"
            "}"
            "QTextEdit QScrollBar::handle:vertical {"
            "  background: rgba(180, 180, 180, 200);"
            "  border-radius: 4px;"
            "}"
        )
        tooltip_edit.setLineWrapMode(QTextEdit.WidgetWidth)
        tooltip_edit.hide()
        return vLine, hLine, tooltip_edit

    def _update_tooltip_position(
        self,
        tooltip_widget: QWidget,
        scene_pos: QPointF,
        view_box: pg.ViewBox,
        margin: int = 10,
    ):
        # 将 scene 坐标映射到 tooltip 父控件（plot_widget）的局部坐标
        plot_widget = tooltip_widget.parentWidget()
        if plot_widget is None:
            plot_widget = view_box.getViewWidget()
        local_pos = plot_widget.mapFromScene(scene_pos)
        tooltip_pos = local_pos

        label_width = tooltip_widget.width()
        label_height = tooltip_widget.height()
        plot_rect = plot_widget.rect()

        # 计算最大可用高度（窗口高度的 80%，防止超出屏幕）
        window = plot_widget.window()
        max_height = int(window.height() * 0.92) if window else plot_rect.height()
        if label_height > max_height:
            label_height = max_height
            tooltip_widget.setFixedHeight(max_height)
        else:
            tooltip_widget.setFixedHeight(label_height)

        # X 方向：优先放在鼠标右侧，超出右边界则放左侧
        if tooltip_pos.x() + label_width > plot_rect.right() - margin:
            x_pos = tooltip_pos.x() - label_width - margin
        else:
            x_pos = tooltip_pos.x() + margin

        # Y 方向：需要同时考虑顶部和底部边界
        space_above = tooltip_pos.y() - plot_rect.top() - margin
        space_below = plot_rect.bottom() - tooltip_pos.y() - margin

        if label_height <= space_above:
            # 上方空间足够，优先放上方（不遮挡鼠标下方内容）
            y_pos = tooltip_pos.y() - label_height - margin
        elif label_height <= space_below:
            # 上方不够但下方够，放下方
            y_pos = tooltip_pos.y() + margin
        else:
            # 上下都不够，选择空间较大的一侧，并做偏移确保可见
            if space_above >= space_below:
                y_pos = plot_rect.top() + margin
            else:
                y_pos = plot_rect.bottom() - label_height - margin

        tooltip_widget.move(int(x_pos), int(y_pos))
        tooltip_widget.raise_()

    def _get_visible_data_slice(
        self, x_min: float, x_max: float, all_data: list
    ) -> list:
        index_min = max(0, int(round(x_min)))
        index_max = min(len(all_data), int(round(x_max)) + 1)

        if index_min >= index_max:
            return [all_data[index_min]] if 0 <= index_min < len(all_data) else []

        return all_data[index_min:index_max]

    def _handle_bounded_x_range_change(
        self,
        view_box: pg.ViewBox,
        x_range: tuple[float, float],
        data_length: int,
        margin: float,
        min_display_range: float,
        y_adapter_func: Callable,
    ):
        x_min, x_max = x_range

        if data_length < min_display_range:
            min_display_range = data_length if data_length > 0 else 1.0

        min_bound = -margin
        max_bound = data_length - 1 + margin
        allowed_range = max_bound - min_bound
        current_range = x_max - x_min

        range_changed = False

        if current_range < min_display_range and data_length > 0:
            center = (x_min + x_max) / 2
            x_min = center - min_display_range / 2
            x_max = center + min_display_range / 2
            current_range = x_max - x_min
            range_changed = True

        if current_range > allowed_range:
            x_min = min_bound
            x_max = max_bound
            current_range = x_max - x_min
            range_changed = True

        if x_min < min_bound:
            x_min = min_bound
            x_max = x_min + current_range
            range_changed = True
        elif x_max > max_bound:
            x_max = max_bound
            x_min = x_max - current_range
            range_changed = True

        if range_changed:
            view_box.setXRange(x_min, x_max, padding=0, update=False)
            x_min, x_max = view_box.viewRange()[0]

        y_adapter_func(view_box, x_min, x_max)

    BENCHMARK_NAME_MAP = {
        "000300.SH": "沪深300",
        "000016.SH": "上证50",
        "000905.SH": "中证500",
        "000852.SH": "中证1000",
        "000001.SH": "上证指数",
        "399001.SZ": "深证成指",
        "399006.SZ": "创业板指",
    }

    def _convert_benchmark_to_klines(self, benchmark_df):
        klines = []
        for dt, row in benchmark_df.iterrows():
            if not isinstance(dt, pd.Timestamp):
                dt = pd.Timestamp(dt)
            bar_dt = dt.to_pydatetime()
            if bar_dt.hour == 0 and bar_dt.minute == 0:
                bar_dt = bar_dt.replace(hour=14, minute=50)
            kline = {
                "datetime": bar_dt,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
            }
            klines.append(kline)
        return klines

    def _create_kline_chart_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        use_benchmark_klines = (
            len(self.result.instruments_data) > 1
            and self.result.benchmark_df is not None
            and not self.result.benchmark_df.empty
            and all(col in self.result.benchmark_df.columns for col in ["open", "high", "low", "close"])
        )

        if use_benchmark_klines:
            klines_full = self._convert_benchmark_to_klines(self.result.benchmark_df)
            benchmark_symbol = self.result.benchmark_symbol or "000300.SH"
            kline_title = f"K线图 - {self.BENCHMARK_NAME_MAP.get(benchmark_symbol, benchmark_symbol)}"
        else:
            required_params = {
                "instrument_id": self.result.strategy_params.get("instrument_id"),
                "exchange": self.result.strategy_params.get("exchange"),
                "kline_style": self.result.strategy_params.get("kline_style"),
            }
            missing_params = [name for name, value in required_params.items() if not value]

            if missing_params:
                missing_str = "、".join(missing_params)
                label = QLabel(f"无法生成K线图，请在回测策略参数中提供：{missing_str}")
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet(
                    "font-size: 18px; color: #d14545; font-weight: bold; margin: 20px;"
                )
                layout.addWidget(label)
                return tab

            if not hasattr(self.result, "klines") or not self.result.klines:
                layout.addWidget(
                    QLabel(
                        "无K线数据，无法生成图表。",
                        alignment=Qt.AlignCenter,
                        styleSheet="QLabel { font-size: 28px; font-weight: bold; }",
                    )
                )
                return tab

            klines_full = self.result.klines[10:]
            kline_title = "K线图"
        if not klines_full:
            label = QLabel("K线数据量不足。", alignment=Qt.AlignCenter)
            label.setStyleSheet("QLabel { font-size: 28px; font-weight: bold; }")
            layout.addWidget(label)
            return tab

        self.all_klines = klines_full
        self.total_data_count = len(klines_full)
        MAX_INITIAL_DISPLAY = 360
        initial_klines = klines_full[-MAX_INITIAL_DISPLAY:]
        self.loaded_data_index = self.total_data_count - len(initial_klines)

        self.kline_data = [
            {
                "time": self.loaded_data_index + i,
                "open": k["open"],
                "high": k["high"],
                "low": k["low"],
                "close": k["close"],
                "datetime": k["datetime"],
            }
            for i, k in enumerate(initial_klines)
        ]
        self.kline_x_axis_ticks = {
            self.loaded_data_index + i: k["datetime"].strftime("%Y-%m-%d %H:%M")
            for i, k in enumerate(initial_klines)
        }

        all_date_strings = [
            k["datetime"].strftime("%Y-%m-%d %H:%M") for k in self.all_klines
        ]
        axis = DateAxis(x_values=all_date_strings, orientation="bottom")
        left_axis = FullValueAxis(orientation="left")
        plot_widget = pg.PlotWidget(
            axisItems={"bottom": axis, "left": left_axis}, useOpenGL=True
        )
        plot_widget.setRenderHint(QPainter.Antialiasing)
        plot_widget.setRenderHint(QPainter.SmoothPixmapTransform)
        plot_widget.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        plot_widget.setCacheMode(QGraphicsView.CacheBackground)
        plot_widget.setBackground("#f7f7f7")
        plot_widget.setTitle(kline_title, color="k", size="16pt", bold=True)
        plot_widget.showGrid(x=True, y=True, alpha=0.3)
        plot_widget.getPlotItem().layout.setContentsMargins(10, 25, 10, 10)

        self.candlestick_item = CandlestickItem(self.kline_data)
        plot_widget.addItem(self.candlestick_item)

        self._add_ma_lines(plot_widget, initial_klines, self.loaded_data_index)

        initial_x_min = self.total_data_count - MAX_INITIAL_DISPLAY
        initial_x_max = self.total_data_count - 0.5
        margin = 5
        initial_x_min = max(0, initial_x_min - margin)
        initial_x_max = initial_x_max + margin

        plot_widget.setXRange(initial_x_min, initial_x_max, padding=0)
        plot_widget.setLabel("left", "价格", **{"color": "k", "font-size": "12pt"})
        plot_widget.setLabel("bottom", "时间", **{"color": "k", "font-size": "12pt"})

        self._add_trade_markers(plot_widget, self.candlestick_item)
        self._add_kline_tooltip(plot_widget)

        view_box = plot_widget.getViewBox()
        view_box.setMouseEnabled(x=True, y=False)
        view_box.enableAutoRange(axis=view_box.YAxis, enable=False)
        self.kline_view_box = view_box

        self._is_initializing = True
        view_box.sigXRangeChanged.connect(self.kline_x_range_changed)
        self.kline_x_range_changed(
            view_box, (initial_x_min, initial_x_max), force_y_only=True
        )
        self._is_initializing = False

        layout.addWidget(plot_widget)
        return tab

    def load_history(self, num_to_load=360):
        if self.loaded_data_index <= 0:
            return False

        start_index = max(0, self.loaded_data_index - num_to_load)
        new_klines = self.all_klines[start_index : self.loaded_data_index]
        if not new_klines:
            self.loaded_data_index = 0
            return False

        new_data_for_plot = [
            {
                "time": start_index + i,
                "open": k["open"],
                "high": k["high"],
                "low": k["low"],
                "close": k["close"],
                "datetime": k["datetime"],
            }
            for i, k in enumerate(new_klines)
        ]
        new_x_axis_ticks = {
            start_index + i: k["datetime"].strftime("%Y-%m-%d %H:%M")
            for i, k in enumerate(new_klines)
        }

        self.kline_data = new_data_for_plot + self.kline_data
        self.kline_x_axis_ticks.update(new_x_axis_ticks)
        self.candlestick_item.data = self.kline_data
        self.candlestick_item.generatePicture()
        self.candlestick_item.update()

        self._update_ma_lines()

        self.loaded_data_index = start_index
        return True

    def _update_ma_lines(self):
        if not hasattr(self, "ma_line_items") or not self.ma_line_items:
            return

        ma_keys = list(self.ma_line_items.keys())

        for ma_key in ma_keys:
            line_item = self.ma_line_items[ma_key]

            ma_data = []
            for i, k in enumerate(self.all_klines[self.loaded_data_index:]):
                if ma_key in k and k[ma_key] is not None:
                    ma_data.append((self.loaded_data_index + i, k[ma_key]))

            if ma_data:
                x_vals = np.array([d[0] for d in ma_data])
                y_vals = np.array([d[1] for d in ma_data])
                line_item.setData(x_vals, y_vals)

    def _add_trade_markers(self, plot_widget, candlestick_item):
        if (
            not hasattr(self, "all_klines")
            or not self.all_klines
            or not self.result.trade_log
        ):
            return

        kline_times = [k["datetime"] for k in self.all_klines]

        trades_by_kline_index = defaultdict(list)
        for trade in self.result.trade_log:
            if trade.trade_price <= 0 or trade.order_id == -1:
                continue
            if not trade.trade_time:
                continue

            idx = bisect.bisect_left(kline_times, trade.trade_time)
            if idx == 0:
                closest_kline_index = 0
            elif idx == len(kline_times):
                closest_kline_index = len(kline_times) - 1
            else:
                closest_kline_index = (
                    idx - 1
                    if abs(trade.trade_time - kline_times[idx - 1])
                    < abs(trade.trade_time - kline_times[idx])
                    else idx
                )
            trades_by_kline_index[closest_kline_index].append(trade)

        (
            buy_spots,
            sell_spots,
            m_spots,
            buy_lines_x,
            buy_lines_y,
            sell_lines_x,
            sell_lines_y,
            m_lines_x,
            m_lines_y,
        ) = ([], [], [], [], [], [], [], [], [])
        price_range = candlestick_item.high_max - candlestick_item.low_min
        arrow_offset, marker_size = price_range * 0.03, 28

        self.marker_arrow_offset = arrow_offset

        marker_font = QFont()
        marker_font.setBold(True)
        marker_font.setPointSize(12)
        multi_trade_color = "#F8AE00"
        buy_color = "#d14545"
        sell_color = "#3f993f"

        for closest_kline_index, trades_list in trades_by_kline_index.items():
            if not trades_list:
                continue
            kline_data = self.all_klines[closest_kline_index]
            num_trades = len(trades_list)
            text_items_for_kline = []

            if num_trades == 1:
                trade = trades_list[0]
                is_buy = trade.direction == "0"
                y_pos = (
                    kline_data["low"] - arrow_offset
                    if is_buy
                    else kline_data["high"] + arrow_offset
                )
                spot_data = {"pos": (closest_kline_index, y_pos), "size": marker_size}

                if y_pos > kline_data["high"]:
                    connect_to_y = kline_data["high"]
                elif y_pos < kline_data["low"]:
                    connect_to_y = kline_data["low"]

                line_coords = (
                    [closest_kline_index, closest_kline_index],
                    [y_pos, connect_to_y],
                )

                if is_buy:
                    buy_spots.append(spot_data)
                    buy_lines_x.extend(line_coords[0])
                    buy_lines_y.extend(line_coords[1])
                else:
                    sell_spots.append(spot_data)
                    sell_lines_x.extend(line_coords[0])
                    sell_lines_y.extend(line_coords[1])

                text = pg.TextItem(
                    text="B" if is_buy else "S", color="#fff8f8", anchor=(0.5, 0.5)
                )
                text.setFont(marker_font)
                text.setPos(closest_kline_index, y_pos)
                text.setZValue(10)
                plot_widget.addItem(text)
                text_items_for_kline.append(text)

            elif num_trades > 1:
                has_buy = any(t.direction == "0" for t in trades_list)
                has_sell = any(t.direction == "1" for t in trades_list)

                marker_text = ""
                text_color = ""
                y_pos = kline_data["high"] + arrow_offset

                if has_buy and has_sell:
                    marker_text = "M"
                    text_color = "#FFFFFF"
                    y_pos = kline_data["high"] + arrow_offset
                elif has_buy:
                    marker_text = "B"
                    text_color = "#fff8f8"
                    y_pos = kline_data["low"] - arrow_offset
                elif has_sell:
                    marker_text = "S"
                    text_color = "#fff8f8"
                    y_pos = kline_data["high"] + arrow_offset

                if marker_text:
                    spot_data = {
                        "pos": (closest_kline_index, y_pos),
                        "size": marker_size,
                    }

                    if y_pos > kline_data["high"]:
                        connect_to_y = kline_data["high"]
                    elif y_pos < kline_data["low"]:
                        connect_to_y = kline_data["low"]
                    else:
                        # y_pos 在 high 和 low 之间，连接到 close 价格
                        connect_to_y = kline_data.get("close", y_pos)

                    line_coords = (
                        [closest_kline_index, closest_kline_index],
                        [y_pos, connect_to_y],
                    )

                    if marker_text == "M":
                        m_spots.append(spot_data)
                        m_lines_x.extend(line_coords[0])
                        m_lines_y.extend(line_coords[1])
                    elif marker_text == "B":
                        buy_spots.append(spot_data)
                        buy_lines_x.extend(line_coords[0])
                        buy_lines_y.extend(line_coords[1])
                    elif marker_text == "S":
                        sell_spots.append(spot_data)
                        sell_lines_x.extend(line_coords[0])
                        sell_lines_y.extend(line_coords[1])

                    text = pg.TextItem(
                        text=marker_text, color=text_color, anchor=(0.5, 0.5)
                    )
                    text.setFont(marker_font)
                    text.setPos(closest_kline_index, y_pos)
                    text.setZValue(10)
                    plot_widget.addItem(text)
                    text_items_for_kline.append(text)

            if text_items_for_kline:
                self.trade_markers[closest_kline_index] = {
                    "items": text_items_for_kline,
                    "trades": trades_list,
                }

        if buy_lines_x:
            plot_widget.addItem(
                pg.PlotDataItem(
                    np.array(buy_lines_x),
                    np.array(buy_lines_y),
                    pen=pg.mkPen(buy_color, width=1, style=Qt.DotLine),
                    connect="pairs",
                )
            )
        if sell_lines_x:
            plot_widget.addItem(
                pg.PlotDataItem(
                    np.array(sell_lines_x),
                    np.array(sell_lines_y),
                    pen=pg.mkPen(sell_color, width=1, style=Qt.DotLine),
                    connect="pairs",
                )
            )
        if m_lines_x:
            plot_widget.addItem(
                pg.PlotDataItem(
                    np.array(m_lines_x),
                    np.array(m_lines_y),
                    pen=pg.mkPen(multi_trade_color, width=1, style=Qt.DotLine),
                    connect="pairs",
                )
            )

        if buy_spots:
            plot_widget.addItem(
                pg.ScatterPlotItem(
                    spots=buy_spots,
                    symbol="o",
                    pen=pg.mkPen(buy_color, width=2),
                    brush=pg.mkBrush(buy_color),
                )
            )
        if sell_spots:
            plot_widget.addItem(
                pg.ScatterPlotItem(
                    spots=sell_spots,
                    symbol="o",
                    pen=pg.mkPen(sell_color, width=2),
                    brush=pg.mkBrush(sell_color),
                )
            )
        if m_spots:
            plot_widget.addItem(
                pg.ScatterPlotItem(
                    spots=m_spots,
                    symbol="o",
                    pen=pg.mkPen(multi_trade_color, width=2),
                    brush=pg.mkBrush(multi_trade_color),
                )
            )

    def _add_ma_lines(self, plot_widget, klines, start_index):
        if not klines:
            return

        ma_keys = [k for k in klines[0].keys() if k.startswith("MA")]
        if not ma_keys:
            return

        self.ma_line_items = {}

        colors = {"MA5": "#FF6B6B", "MA10": "#4ECDC4", "MA20": "#45B7D1", "MA30": "#96CEB4", "MA60": "#FFEAA7"}

        for ma_key in ma_keys:
            ma_data = []
            for i, k in enumerate(self.all_klines):
                if ma_key in k and k[ma_key] is not None:
                    ma_data.append((i, k[ma_key]))

            if ma_data:
                x_vals = np.array([d[0] for d in ma_data])
                y_vals = np.array([d[1] for d in ma_data])
                color = colors.get(ma_key, "#888888")
                line_item = pg.PlotDataItem(
                    x_vals,
                    y_vals,
                    pen=pg.mkPen(color, width=1.5),
                    name=ma_key,
                )
                plot_widget.addItem(line_item)
                self.ma_line_items[ma_key] = line_item

    def _add_kline_tooltip(self, plot_widget):
        self.kline_vLine, self.kline_hLine, self.kline_tooltip_label = (
            self._create_crosshair_items(plot_widget)
        )
        self._kline_tooltip_pinned = False
        self.kline_proxy = pg.SignalProxy(
            plot_widget.scene().sigMouseMoved, rateLimit=30, slot=self.kline_mouse_moved
        )
        plot_widget.scene().sigMouseClicked.connect(self._kline_scene_clicked)

    def _kline_scene_clicked(self, event):
        if not hasattr(self, "kline_tooltip_label"):
            return
        if self._kline_tooltip_pinned:
            self._kline_tooltip_pinned = False
            self.kline_tooltip_label.setStyleSheet(
                "QTextEdit {"
                "  background-color: rgba(255, 255, 255, 235);"
                "  border: 1px solid #cccccc;"
                "  padding: 6px;"
                "  font-size: 15px;"
                "  font-family: Microsoft YaHei;"
                "}"
                "QTextEdit QScrollBar:vertical {"
                "  width: 8px;"
                "  background: rgba(240, 240, 240, 200);"
                "}"
                "QTextEdit QScrollBar::handle:vertical {"
                "  background: rgba(180, 180, 180, 200);"
                "  border-radius: 4px;"
                "}"
            )
            self.kline_tooltip_label.hide()
            self.kline_vLine.hide()
            self.kline_hLine.hide()
            return
        if self.kline_tooltip_label.isVisible():
            self._kline_tooltip_pinned = True
            self.kline_tooltip_label.setStyleSheet(
                "QTextEdit {"
                "  background-color: rgba(255, 255, 255, 245);"
                "  border: 2px solid #4a90d9;"
                "  padding: 6px;"
                "  font-size: 15px;"
                "  font-family: Microsoft YaHei;"
                "}"
                "QTextEdit QScrollBar:vertical {"
                "  width: 8px;"
                "  background: rgba(240, 240, 240, 200);"
                "}"
                "QTextEdit QScrollBar::handle:vertical {"
                "  background: rgba(180, 180, 180, 200);"
                "  border-radius: 4px;"
                "}"
            )

    def kline_mouse_moved(self, event):
        if not hasattr(self, "kline_vLine") or not hasattr(self, "kline_data"):
            return
        if self._kline_tooltip_pinned:
            return
        pos = event[0]
        vb = self.kline_vLine.getViewBox()
        if vb.sceneBoundingRect().contains(pos):
            mousePoint = vb.mapSceneToView(pos)
            x, y = mousePoint.x(), mousePoint.y()
            kline_times = [k["time"] for k in self.kline_data]
            idx = bisect.bisect_left(kline_times, x)
            closest_kline, min_distance = None, float("inf")
            indices_to_check = [idx - 1] if idx > 0 else []
            if idx < len(self.kline_data):
                indices_to_check.append(idx)
            if not indices_to_check:
                return

            for i in indices_to_check:
                distance = abs(self.kline_data[i]["time"] - x)
                if distance < min_distance:
                    min_distance = distance
                    closest_kline = self.kline_data[i]

            if closest_kline and min_distance < 0.5:
                self.kline_vLine.show()
                self.kline_hLine.show()
                self.kline_vLine.setPos(x)
                self.kline_hLine.setPos(y)

                kline = closest_kline
                datetime_str = self.kline_x_axis_ticks.get(kline["time"], "未知时间")

                kline_parts = [
                    f'<span style="font-size: 16px;"><b>{kline["time"]}</b></span>',
                    f'<span style="font-size: 15px;"><b>时间</b>: {datetime_str}</span>',
                    f'<span style="font-size: 15px;"><b>开盘</b>: <span style="color: #000000;">{kline["open"]:.2f}</span></span>',
                    f'<span style="font-size: 15px;"><b>收盘</b>: <span style="color: #000000;">{kline["close"]:.2f}</span></span>',
                    f'<span style="font-size: 15px;"><b>最高</b>: <span style="color: #000000;">{kline["high"]:.2f}</span></span>',
                    f'<span style="font-size: 15px;"><b>最低</b>: <span style="color: #000000;">{kline["low"]:.2f}</span></span>',
                ]

                trade_parts = []
                trade_list = self.trade_markers.get(kline["time"], {}).get("trades")

                if trade_list:
                    for trade_info in trade_list:
                        if trade_info.direction == "0":
                            color = "#d14545"
                            action = "买开" if trade_info.offset == "0" else "买平"
                        else:
                            color = "#3f993f"
                            action = "卖开" if trade_info.offset == "0" else "卖平"

                        trade_text = f"{trade_info.instrument_id} {action} {trade_info.volume} 手 @ {trade_info.trade_price:.2f}"
                        trade_parts.append(
                            f'<span style="color: {color}; font-size: 15px;"><strong>{trade_text}</strong></span>'
                        )

                final_html_parts = ["<br>".join(kline_parts)]
                if trade_parts:
                    final_html_parts.append(
                        '<div style="margin: 3px 0; padding: 0; height: 1px; border-top: 1px dashed #999;"></div>'
                    )
                    final_html_parts.append("<br>".join(trade_parts))

                self.kline_tooltip_label.setHtml("".join(final_html_parts))

                self.kline_tooltip_label.adjustSize()
                self._update_tooltip_position(self.kline_tooltip_label, pos, vb)
                self.kline_tooltip_label.show()
            else:
                self.kline_tooltip_label.hide()
                self.kline_vLine.hide()
                self.kline_hLine.hide()
        else:
            self.kline_tooltip_label.hide()
            self.kline_vLine.hide()
            self.kline_hLine.hide()

    def kline_x_range_changed(self, view_box, x_range, force_y_only=False):
        if not hasattr(self, "kline_data") or not self.kline_data:
            return
        if (
            hasattr(self, "_is_initializing")
            and self._is_initializing
            and not force_y_only
        ):
            if force_y_only:
                self._synchronize_markers(*view_box.viewRange()[0])
            return

        x_min, x_max = x_range
        if not force_y_only:
            total_data_count = len(self.all_klines)
            current_range = x_max - x_min
            min_bound, max_bound = -0.5 - 5, total_data_count - 0.5 + 5
            allowed_range = max_bound - min_bound if total_data_count > 0 else 1.0
            range_changed = False

            min_display_range = 3.0
            limit_range = 1440.0
            range_limit_applied = False

            if (
                current_range < min_display_range
                and total_data_count > min_display_range
            ):
                center_x = (x_min + x_max) / 2
                x_min = center_x - min_display_range / 2
                x_max = center_x + min_display_range / 2
                current_range = x_max - x_min
                range_limit_applied = True
                range_changed = True

            elif current_range > limit_range:
                center_x = (x_min + x_max) / 2
                x_min = center_x - (limit_range / 2)
                x_max = center_x + (limit_range / 2)
                current_range = x_max - x_min
                range_limit_applied = True
                range_changed = True

            if x_max > max_bound:
                shift = x_max - max_bound
                x_max = max_bound
                x_min = x_min - shift
                range_changed = True
            elif x_min < min_bound:
                shift = min_bound - x_min
                x_min = min_bound
                x_max = x_max + shift
                range_changed = True

            if range_changed:
                if x_min < min_bound:
                    x_min = min_bound
                    x_max = x_min + current_range
                if x_max > max_bound:
                    x_max = max_bound
                    x_min = x_max - current_range

                view_box.setXRange(x_min, x_max, padding=0, update=False)
                x_min, x_max = view_box.viewRange()[0]

            if (
                x_min < self.loaded_data_index + (x_max - x_min) * 0.1
                and self.loaded_data_index > 0
            ):
                if self.load_history():
                    current_x_min, current_x_max = view_box.viewRange()[0]
                    current_center = (current_x_min + current_x_max) / 2
                    current_range_after_load = current_x_max - current_x_min

                    new_x_min = current_center - current_range_after_load / 2
                    new_x_max = current_center + current_range_after_load / 2

                    new_min_bound = -0.5 - 5
                    if new_x_min < new_min_bound:
                        shift = new_min_bound - new_x_min
                        new_x_min = new_min_bound
                        new_x_max += shift

                    view_box.setXRange(new_x_min, new_x_max, padding=0, update=False)
                    x_min, x_max = view_box.viewRange()[0]

        self._synchronize_markers(x_min, x_max)
        self._adapt_kline_y_axis(view_box, x_min, x_max)

    def _adapt_kline_y_axis(self, view_box, x_min, x_max):
        visible_data = [
            d for d in self.kline_data if int(x_min) <= d["time"] < int(x_max) + 1
        ]
        if not visible_data or not hasattr(self, "candlestick_item"):
            return

        # 过滤掉 NaN 值
        valid_lows = [d["low"] for d in visible_data if not (d["low"] != d["low"])]  # NaN check
        valid_highs = [d["high"] for d in visible_data if not (d["high"] != d["high"])]  # NaN check

        if not valid_lows or not valid_highs:
            return

        kline_y_min, kline_y_max = min(valid_lows), max(valid_highs)

        marker_offset = getattr(self, "marker_arrow_offset", 0.0)

        final_y_min = kline_y_min - marker_offset
        final_y_max = kline_y_max + marker_offset

        total_range = self.candlestick_item.high_max - self.candlestick_item.low_min
        min_padding = total_range * 0.025

        visible_range = final_y_max - final_y_min
        dyn_padding = visible_range * 0.05

        padding = (
            max(min_padding, dyn_padding)
            if visible_range > 0
            else max(0.005, min_padding)
        )

        view_box.setYRange(final_y_min - padding, final_y_max + padding, padding=0)

    def _synchronize_markers(self, x_min: float, x_max: float):
        if not self.trade_markers:
            return
        for index, data in self.trade_markers.items():
            is_visible = x_min - 0.5 <= index <= x_max + 0.5
            for item in data["items"]:
                if item.isVisible() != is_visible:
                    item.setVisible(is_visible)

    def _create_overview_tab(self) -> QWidget:
        tab = QWidget()
        layout = QGridLayout(tab)
        layout.addWidget(self._create_metrics_panel(), 0, 0)
        self._create_equity_chart_panel(layout)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 3)
        return tab

    def _add_metric_separator(self, layout: QGridLayout, row: int, title: str) -> int:
        title_label = QLabel(title)
        title_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        layout.addWidget(title_label, row, 0, 1, 2)
        return row + 1

    def _create_metrics_panel(self) -> QWidget:
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setStyleSheet(
            "QFrame { background-color: #f7f7f7; border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; } QLabel { background-color: transparent; border: none; padding: 3px; }"
        )
        layout = QGridLayout(panel)
        layout.setSpacing(10)
        all_metrics = {
            "整体表现": {
                "初始资金": f"{self.result.account.initial_capital:,.2f}",
                "结束资金": f"{self.result.account.dynamic_rights:,.2f}",
                "总收益": f"{self.result.account.total_profit:,.2f}",
                "收益率": f"{self.result.account.rate:.2%}",
                "年化收益率": f"{self.result.annual_return(self.result.account.initial_capital, self.result.account.dynamic_rights):.2%}",
            },
            "风险评估": {
                "最大回撤": f"{self.result.max_drawdown():.2%}",
                "夏普比率": f"{self.result.sharpe_ratio():.2f}",
            },
            "交易统计": {
                "总成交额": f"{self.result.turnover:,.2f}",
                "总手续费": f"{self.result.account.fee:,.2f}",
                "总成交手数": f"{self.result.total_volume}",
                "总交易日": f"{self.result.total_trading_days}",
                "盈利天数": f"{self.result.pnl_days[0]}",
                "亏损天数": f"{self.result.pnl_days[1]}",
            },
        }

        print("\n" + "="*60)
        print("回测结果汇总")
        print("="*60)
        for group_title, metrics in all_metrics.items():
            print(f"\n【{group_title}】")
            for name, value in metrics.items():
                print(f"  {name}: {value}")
        print("="*60 + "\n")

        row = 0
        for group_title, metrics in all_metrics.items():
            row = self._add_metric_separator(layout, row, group_title)
            for name, value in metrics.items():
                name_label, value_label = QLabel(name), QLabel(value)
                name_label.setFont(QFont("Microsoft YaHei", 11))
                name_label.setStyleSheet("color: #666;")
                value_label.setFont(QFont("Arial", 11, QFont.Bold))
                value_label.setAlignment(Qt.AlignRight)
                if "收益" in name or "盈亏" in name:
                    value_label.setStyleSheet(
                        "color: #d14545;"
                        if self.result.account.total_profit > 0
                        else "color: #3f993f;"
                    )
                layout.addWidget(name_label, row, 0)
                layout.addWidget(value_label, row, 1)
                row += 1
            row += 1
        layout.addItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding),
            row,
            0,
            1,
            2,
        )
        return panel

    def _create_equity_chart_panel(self, parent_layout: QGridLayout):
        df = self.plot_df
        if df is None or df.empty:
            return
        date_strings = df["datetime"].dt.strftime("%Y-%m-%d").tolist()
        bottom_axis, left_axis = (
            DateAxis(x_values=date_strings, orientation="bottom"),
            FullValueAxis(orientation="left"),
        )
        if 0 < len(date_strings) <= 3:
            bottom_axis.setTicks([[(i, date) for i, date in enumerate(date_strings)]])

        plot_widget = pg.PlotWidget(
            axisItems={"bottom": bottom_axis, "left": left_axis}
        )
        plot_widget.setBackground("#f7f7f7")
        plot_widget.showGrid(x=True, y=True, alpha=0.3)
        plot_widget.setTitle("资金曲线", color="k", size="16pt", bold=True)
        plot_widget.setLabel("left", "权益 (元)", **{"color": "k", "font-size": "12pt"})
        plot_widget.getPlotItem().layout.setContentsMargins(10, 25, 10, 10)

        x_axis_data, equity_curve_data = (
            list(range(len(df.index))),
            df["PortfolioValue"].tolist(),
        )
        self.plot_curve = plot_widget.plot(
            x_axis_data,
            equity_curve_data,
            pen=pg.mkPen("#007bff", width=2.5, style=Qt.SolidLine),
            name="资金曲线",
        )

        if not df.empty:
            max_idx, min_idx = (
                df["PortfolioValue"].idxmax(),
                df["PortfolioValue"].idxmin(),
            )
            max_val, min_val = (
                df.loc[max_idx, "PortfolioValue"],
                df.loc[min_idx, "PortfolioValue"],
            )
            max_pos, min_pos = df.index.get_loc(max_idx), df.index.get_loc(min_idx)
            max_text = pg.TextItem(
                f"峰值: {max_val:,.2f}",
                anchor=(0.5, 1.5),
                color="#d14545",
                fill=pg.mkBrush(255, 255, 255, 150),
            )
            max_text.setPos(max_pos, max_val)
            plot_widget.addItem(max_text)
            min_text = pg.TextItem(
                f"谷值: {min_val:,.2f}",
                anchor=(0.5, -0.5),
                color="#3f993f",
                fill=pg.mkBrush(255, 255, 255, 150),
            )
            min_text.setPos(min_pos, min_val)
            plot_widget.addItem(min_text)

        view_box = plot_widget.getViewBox()
        view_box.setMouseEnabled(x=True, y=False)
        margin = 2
        initial_x_min = -margin if not equity_curve_data else -margin
        initial_x_max = (
            margin if not equity_curve_data else len(equity_curve_data) - 1 + margin
        )
        plot_widget.setXRange(initial_x_min, initial_x_max, padding=0)
        self._adapt_equity_y_axis(
            view_box, initial_x_min, initial_x_max, equity_curve_data
        )

        y_adapter = lambda vb, x_min, x_max: self._adapt_equity_y_axis(
            vb, x_min, x_max, equity_curve_data
        )
        view_box.sigXRangeChanged.connect(
            lambda vb, x_range: self._handle_bounded_x_range_change(
                vb,
                x_range,
                len(equity_curve_data),
                margin=2,
                min_display_range=3.0,
                y_adapter_func=y_adapter,
            )
        )

        self.vLine, self.hLine, self.info_label = self._create_crosshair_items(
            plot_widget
        )
        self.proxy = pg.SignalProxy(
            plot_widget.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved
        )
        parent_layout.addWidget(plot_widget, 0, 1)

    def _adapt_equity_y_axis(self, view_box, x_min, x_max, equity_data):
        visible_equity = self._get_visible_data_slice(x_min, x_max, equity_data)
        if not visible_equity:
            return
        y_min, y_max = min(visible_equity), max(visible_equity)
        equity_range = y_max - y_min
        padding = (
            (y_min * 0.1 if y_min != 0 else 1)
            if equity_range <= 0
            else equity_range * 0.1
        )
        view_box.setYRange(y_min - padding, y_max + padding, padding=0)

    def mouse_moved(self, event):
        pos = event[0]
        if self.plot_curve is None or not self.plot_curve.isVisible():
            return
        vb = self.plot_curve.getViewBox()
        if vb.sceneBoundingRect().contains(pos):
            mousePoint = vb.mapSceneToView(pos)
            index = int(round(mousePoint.x())) if not np.isnan(mousePoint.x()) else -1
            if 0 <= index < len(self.result.df.index):
                self.vLine.show()
                self.hLine.show()
                date_str = self.result.df.iloc[index]["datetime"].strftime("%Y-%m-%d")
                equity_value = self.result.df["PortfolioValue"].iloc[index]
                self.vLine.setPos(index)
                self.hLine.setPos(mousePoint.y())
                self.info_label.setPlainText(f"日期: {date_str}\n权益: {equity_value:,.2f}")
                self.info_label.adjustSize()
                self._update_tooltip_position(self.info_label, pos, vb)
                self.info_label.show()
            else:
                self.info_label.hide()
                self.vLine.hide()
                self.hLine.hide()
        else:
            self.info_label.hide()
            self.vLine.hide()
            self.hLine.hide()

    def _create_daily_pnl_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 20, 10, 10)
        df = self.plot_df
        if df is None or df.empty:
            layout.addWidget(
                QLabel(
                    "无有效交易日数据",
                    alignment=Qt.AlignCenter,
                    styleSheet="QLabel { font-size: 28px; font-weight: bold; }",
                )
            )
            return tab

        date_strings = df["datetime"].dt.strftime("%Y-%m-%d").tolist()
        axis = DateAxis(x_values=date_strings, orientation="bottom")
        if 0 < len(date_strings) <= 3:
            axis.setTicks([[(i, date) for i, date in enumerate(date_strings)]])

        plot_widget = pg.PlotWidget(axisItems={"bottom": axis})
        plot_widget.setTitle("每日收益（含手续费）", color="k", size="16pt", bold=True)
        plot_widget.setBackground("#f7f7f7")
        plot_widget.showGrid(x=True, y=True, alpha=0.3)
        plot_widget.getPlotItem().layout.setContentsMargins(10, 25, 10, 10)
        view_box = plot_widget.getViewBox()
        view_box.setMouseEnabled(x=True, y=False)

        x, y = list(range(len(df.index))), df["PnL"].values.tolist()
        data_length, margin = len(y), 1
        initial_x_min = -margin if not y else -margin
        initial_x_max = margin if not y else data_length - 1 + margin
        plot_widget.setXRange(initial_x_min, initial_x_max, padding=0)
        self._adapt_pnl_y_axis(view_box, initial_x_min, initial_x_max, y)

        y_adapter = lambda vb, x_min, x_max: self._adapt_pnl_y_axis(vb, x_min, x_max, y)
        view_box.sigXRangeChanged.connect(
            lambda vb, x_range: self._handle_bounded_x_range_change(
                vb, x_range, data_length, margin, 3.0, y_adapter
            )
        )

        plot_widget.addItem(
            pg.BarGraphItem(
                x=x,
                height=y,
                width=0.6,
                brushes=["#d14545" if val >= 0 else "#3f993f" for val in y],
            )
        )

        vLine, hLine, pnl_info_label = self._create_crosshair_items(plot_widget)

        def mouse_moved_pnl(event):
            pos, vb = event, plot_widget.getViewBox()
            if vb.sceneBoundingRect().contains(pos):
                index = int(vb.mapSceneToView(pos).x() + 0.5)
                if 0 <= index < len(date_strings):
                    vLine.show()
                    hLine.show()
                    vLine.setPos(index)
                    hLine.setPos(y[index])
                    pnl_info_label.setPlainText(
                        f"日期: {date_strings[index]}\n收益: {y[index]:,.2f}"
                    )
                    pnl_info_label.adjustSize()
                    self._update_tooltip_position(pnl_info_label, pos, vb)
                    pnl_info_label.show()
                else:
                    pnl_info_label.hide()
                    vLine.hide()
                    hLine.hide()
            else:
                pnl_info_label.hide()
                vLine.hide()
                hLine.hide()

        plot_widget.scene().sigMouseMoved.connect(mouse_moved_pnl)
        plot_widget.setLabel("left", "盈亏 (元)", **{"color": "k", "font-size": "12pt"})
        layout.addWidget(plot_widget)
        return tab

    def _adapt_pnl_y_axis(self, view_box, x_min, x_max, pnl_data):
        visible_pnl = self._get_visible_data_slice(x_min, x_max, pnl_data)
        if not visible_pnl:
            return
        y_min, y_max = min(visible_pnl), max(visible_pnl)
        if y_min >= 0:
            padding = y_max * 0.05 if y_max > 0 else 0.1
            y_min, y_max = max(0, y_min - padding), y_max + padding
        elif y_max <= 0:
            padding = abs(y_min) * 0.05 if y_min < 0 else 0.1
            y_min, y_max = y_min - padding, min(0, y_max + padding)
        else:
            y_min -= abs(y_min) * 0.05
            y_max += y_max * 0.05
        view_box.setYRange(y_min, y_max, padding=0)

    def _process_trades_for_pnl(self) -> list:
        processed_list, positions = [], {}
        multipliers = {
            iid: data.volume_multiple
            for iid, data in self.result.instruments_data.items()
        }
        for trade in self.result.trade_log:
            instrument, is_opening, is_buy = (
                trade.instrument_id,
                trade.offset == "0",
                trade.direction == "0",
            )
            multiplier = multipliers.get(instrument, 1)
            trade_data = {
                "order_id": trade.order_id,
                "time": trade.trade_time,
                "instrument": instrument,
                "volume": trade.volume,
                "order_price": trade.order_price,
                "price": trade.trade_price,
                "turnover": trade.trade_price * abs(trade.volume) * multiplier,
                "pnl": 0.0,
                "fee": getattr(trade, "fee", 0.0),
                "memo": trade.memo,
            }
            is_traded = trade.trade_price > 0 and trade.order_id != -1

            if is_opening:
                action = "买开" if is_buy else "卖开"
                if is_traded:
                    positions.setdefault(instrument, deque()).append(
                        {"price": trade.trade_price, "volume": trade.volume}
                    )
            else:
                action = "买平" if is_buy else "卖平"
                if is_traded and instrument in positions and positions[instrument]:
                    pnl, volume_to_close, open_positions = (
                        0.0,
                        abs(trade.volume),
                        positions[instrument],
                    )
                    while volume_to_close > 0 and open_positions:
                        oldest_open = open_positions[0]
                        match_volume = min(volume_to_close, oldest_open["volume"])
                        price_diff = trade.trade_price - oldest_open["price"]
                        pnl += (
                            (price_diff if not is_buy else -price_diff)
                            * match_volume
                            * multipliers.get(instrument, 1)
                        )
                        volume_to_close -= match_volume
                        oldest_open["volume"] -= match_volume
                        if oldest_open["volume"] == 0:
                            open_positions.popleft()
                    trade_data["pnl"] = pnl
            trade_data["action"] = action if "action" in locals() else "未知"
            processed_list.append(trade_data)
        return processed_list

    def _create_trade_stats_panel(self, processed_trades: list) -> QWidget:
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setStyleSheet(
            "QFrame { background-color: #f7f7f7; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; } QLabel { background-color: transparent; border: none; padding: 3px; }"
        )
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)

        checkbox_layout = QHBoxLayout()
        checkbox_layout.setSpacing(20)

        self.show_untraded_checkbox = QCheckBox("显示未成交报单")
        self.show_untraded_checkbox.setChecked(True)
        self.show_untraded_checkbox.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.show_untraded_checkbox.setStyleSheet("""
            QCheckBox { 
                padding: 5px 10px;
                min-height: 30px;
            }
        """)
        self.show_untraded_checkbox.stateChanged.connect(self._update_trade_table)
        checkbox_layout.addWidget(self.show_untraded_checkbox)

        self.instrument_toggle_button = QPushButton("隐藏合约")
        self.instrument_toggle_button.setFont(QFont("Microsoft YaHei", 12))
        self.instrument_toggle_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 5px 15px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        self.instrument_toggle_button.clicked.connect(self._toggle_instrument_visibility)
        checkbox_layout.addWidget(self.instrument_toggle_button)

        layout.addLayout(checkbox_layout)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #e0e0e0; margin: 8px 0;")
        layout.addWidget(line)
        stats_layout = QGridLayout()
        stats_layout.setVerticalSpacing(8)
        stats_layout.setHorizontalSpacing(15)
        # 设置第一列（名称列）的最小宽度，确保文字完整显示
        stats_layout.setColumnMinimumWidth(0, 150)
        pnl_values = [
            t["pnl"]
            for t in processed_trades
            if isinstance(t["pnl"], (int, float)) and t["pnl"] != 0.0
        ]
        total_trades = len(pnl_values)

        profits, losses = [], []
        if total_trades > 0:
            profits, losses = (
                [p for p in pnl_values if p > 0],
                [p for p in pnl_values if p < 0],
            )
            win_count, loss_count = len(profits), len(losses)
            win_rate = win_count / total_trades if total_trades > 0 else 0
            total_profit, total_loss = sum(profits), sum(losses)
            avg_profit = total_profit / win_count if win_count > 0 else 0
            avg_loss = total_loss / loss_count if loss_count > 0 else 0

            profit_factor = (
                "∞"
                if total_loss == 0
                else (
                    "0.00"
                    if win_count == 0
                    else f"{abs((total_profit/win_count) / (total_loss/loss_count)):.2f}"
                )
            )
            avg_pnl = sum(pnl_values) / total_trades

            max_consecutive_wins = 0
            max_consecutive_losses = 0
            current_wins = 0
            current_losses = 0

            closing_trades_pnl = [t["pnl"] for t in processed_trades if t["pnl"] != 0.0]

            for pnl in closing_trades_pnl:
                if pnl > 0:
                    current_wins += 1
                    current_losses = 0
                elif pnl < 0:
                    current_losses += 1
                    current_wins = 0

                max_consecutive_wins = max(max_consecutive_wins, current_wins)
                max_consecutive_losses = max(max_consecutive_losses, current_losses)

            avg_win_hold_time, avg_loss_hold_time = self._calc_avg_hold_time(processed_trades)

            risk_reward_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0

            expectancy = (win_rate * avg_profit + (1 - win_rate) * avg_loss) if total_trades > 0 else 0

            profit_factor_calc = abs(total_profit / total_loss) if total_loss != 0 else 0

            sharpe_approx = self.result.sharpe_ratio()

            win_rate_stability = "稳定" if max_consecutive_losses <= 3 else "一般" if max_consecutive_losses <= 5 else "差"

        else:
            (
                avg_pnl,
                win_rate,
                total_profit,
                total_loss,
                profit_factor,
                win_count,
                loss_count,
                avg_profit,
                avg_loss,
                max_consecutive_wins,
                max_consecutive_losses,
                avg_win_hold_time,
                avg_loss_hold_time,
                risk_reward_ratio,
                expectancy,
                profit_factor_calc,
                sharpe_approx,
                win_rate_stability,
            ) = (0, 0, 0, 0, "N/A", 0, 0, 0, 0, 0, 0, "N/A", "N/A", 0, 0, 0, 0, "N/A")

        stats = {
            "平仓交易次数": f"{total_trades}",
            "胜率": f"{win_rate:.2%}",
            "盈利次数": f"{win_count}",
            "亏损次数": f"{loss_count}",
            "持平次数": f"{total_trades - win_count - loss_count}",
            "最大持续盈利次数": f"{max_consecutive_wins}",
            "最大持续亏损次数": f"{max_consecutive_losses}",
            "胜率稳定性": win_rate_stability,
            "": "",
            "总盈利": f"{total_profit:,.2f}",
            "总亏损": f"{total_loss:,.2f}",
            "净盈亏": f"{total_profit + total_loss:,.2f}",
            "平均每次盈亏": f"{avg_pnl:,.2f}",
            "盈亏比": profit_factor,
            "收益风险比": f"{risk_reward_ratio:.2f}",
            "期望值": f"{expectancy:,.2f}",
            "夏普比率(近似)": f"{sharpe_approx:.2f}",
            " ": "",
            "平均每次盈利": f"{avg_profit:,.2f}",
            "平均每次亏损": f"{avg_loss:,.2f}",
            "单次最大盈利": f"{max(profits):,.2f}" if profits else "N/A",
            "单次最大亏损": f"{min(losses):,.2f}" if losses else "N/A",
            "  ": "",
            "盈利交易平均持仓": avg_win_hold_time if total_trades > 0 else "N/A",
            "亏损交易平均持仓": avg_loss_hold_time if total_trades > 0 else "N/A",
        }
        row = 0
        for name, value in stats.items():
            name_label, value_label = QLabel(name), QLabel(value)
            name_label.setFont(QFont("Microsoft YaHei", 10))
            name_label.setStyleSheet("color: #666;")
            value_label.setFont(QFont("Arial", 10, QFont.Bold))
            value_label.setAlignment(Qt.AlignRight)
            stats_layout.addWidget(name_label, row, 0)
            stats_layout.addWidget(value_label, row, 1)
            row += 1
            if name == "最大持续亏损次数":
                spacer = QSpacerItem(20, 30, QSizePolicy.Minimum, QSizePolicy.Fixed)
                stats_layout.addItem(spacer, row, 0, 1, 2)
                row += 1
        layout.addLayout(stats_layout)
        layout.addStretch()
        panel.setMinimumWidth(320)
        panel.setMinimumHeight(700)
        return panel

    def _calc_avg_hold_time(self, processed_trades: list) -> tuple:
        from datetime import datetime

        open_trades = {}
        win_hold_times = []
        loss_hold_times = []

        for trade in processed_trades:
            action = trade.get("action", "")
            time_str = trade.get("time", "")
            pnl = trade.get("pnl", 0)

            if not time_str:
                continue

            try:
                if isinstance(time_str, str):
                    dt = datetime.strptime(time_str, "%Y%m%d %H:%M:%S")
                else:
                    dt = time_str
            except:
                continue

            if "开" in action:
                key = f"{trade.get('instrument', '')}_{trade.get('order_id', '')}"
                open_trades[key] = dt
            elif "平" in action and pnl != 0:
                if open_trades:
                    open_dt = list(open_trades.values())[-1]
                    hold_time = (dt - open_dt).total_seconds() / 60

                    if pnl > 0:
                        win_hold_times.append(hold_time)
                    elif pnl < 0:
                        loss_hold_times.append(hold_time)

                    if open_trades:
                        open_trades.popitem()

        avg_win = f"{sum(win_hold_times)/len(win_hold_times):.1f}分钟" if win_hold_times else "N/A"
        avg_loss = f"{sum(loss_hold_times)/len(loss_hold_times):.1f}分钟" if loss_hold_times else "N/A"

        return avg_win, avg_loss

    def _create_trade_log_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        self.all_processed_trades = self._process_trades_for_pnl()
        stats_panel = self._create_trade_stats_panel(self.all_processed_trades)
        layout.addWidget(stats_panel)
        self.trade_table = QTableWidget()
        self._setup_trade_table()
        self._populate_trade_table(self.all_processed_trades)
        layout.addWidget(self.trade_table)
        layout.setStretchFactor(stats_panel, 1)
        layout.setStretchFactor(self.trade_table, 3)
        return tab

    def _setup_trade_table(self):
        headers = [
            "报单号",
            "时间",
            "标的",
            "交易",
            "数量",
            "报单价",
            "成交价",
            "成交额",
            "盈亏",
            "手续费",
            "备注",
        ]
        self.trade_table.setColumnCount(len(headers))
        self.trade_table.setHorizontalHeaderLabels(headers)
        self.trade_table.setSortingEnabled(True)
        self.trade_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.trade_table.verticalHeader().setVisible(False)

        header = self.trade_table.horizontalHeader()
        header.setSectionsMovable(False)

        for col in range(len(headers)):
            header.setSectionResizeMode(col, QHeaderView.Interactive)

        header.sectionResized.connect(self._on_column_resized)
        header.sectionClicked.connect(self._on_header_clicked)

        self.trade_table.model().dataChanged.connect(self._on_data_changed)

        self._setup_filter_buttons()

    def _on_column_resized(self, logical_index, old_size, new_size):
        self._user_adjusted_columns.add(logical_index)
        self._column_widths[logical_index] = new_size

    def _on_data_changed(self, top_left, bottom_right, roles):
        for col in range(top_left.column(), bottom_right.column() + 1):
            if col not in self._user_adjusted_columns:
                self._adjust_column_width(col)

    def _adjust_column_width(self, col):
        header = self.trade_table.horizontalHeader()
        max_width = 0
        font_metrics = self.trade_table.fontMetrics()

        header_text = self.trade_table.horizontalHeaderItem(col).text()
        header_width = font_metrics.width(header_text) + 40
        max_width = max(max_width, header_width)

        for row in range(self.trade_table.rowCount()):
            item = self.trade_table.item(row, col)
            if item:
                text = item.text()
                text_width = font_metrics.width(text) + 30
                max_width = max(max_width, text_width)

        max_width = min(max_width, 300)
        max_width = max(max_width, 80)

        visual_margin = 20
        final_width = max_width + visual_margin

        if col not in self._user_adjusted_columns:
            header.resizeSection(col, final_width)
            self._column_widths[col] = final_width

    def _auto_adjust_all_columns(self):
        for col in range(self.trade_table.columnCount()):
            if col not in self._user_adjusted_columns:
                self._adjust_column_width(col)

    def _on_header_clicked(self, logical_index):
        if self._instrument_hidden:
            QMessageBox.warning(
                self,
                "提示",
                "无法在隐藏状态下进行排序操作"
            )

    def _setup_filter_buttons(self):
        header = self.trade_table.horizontalHeader()
        for col in range(self.trade_table.columnCount()):
            btn = FilterButton(header)
            btn.clicked.connect(lambda checked, c=col: self._show_filter_menu(c))
            self._filter_buttons[col] = btn
        self._update_filter_button_positions()

    def _update_filter_button_positions(self):
        header = self.trade_table.horizontalHeader()
        for col, btn in self._filter_buttons.items():
            x = header.sectionPosition(col) + header.sectionSize(col) - 18
            y = (header.height() - 16) // 2
            btn.move(x, y)
            btn.setVisible(header.sectionSize(col) > 40)

    def _show_filter_menu(self, col):
        menu = QMenu(self)
        menu.setStyleSheet("QMenu { font-family: Microsoft YaHei; font-size: 12px; }")

        values = set()
        for row in range(self.trade_table.rowCount()):
            item = self.trade_table.item(row, col)
            if item:
                values.add(item.text())

        sorted_values = sorted(values, key=lambda x: (x == "--", x))

        search_edit = QLineEdit(menu)
        search_edit.setPlaceholderText("搜索...")
        search_edit.setClearButtonEnabled(True)
        search_edit.setStyleSheet("QLineEdit { padding: 4px; }")
        search_action = QWidgetAction(menu)
        search_action.setDefaultWidget(search_edit)
        menu.addAction(search_action)
        menu.addSeparator()

        select_all_action = QAction("(全选)", menu)
        menu.addAction(select_all_action)

        actions = []
        for val in sorted_values:
            action = QAction(val, menu)
            action.setCheckable(True)
            current_filter = self._column_filters.get(col)
            if current_filter is None:
                action.setChecked(True)
            else:
                action.setChecked(val in current_filter)
            menu.addAction(action)
            actions.append(action)

        select_all_action.triggered.connect(lambda: self._toggle_all_filter(actions))
        search_edit.textChanged.connect(lambda text: self._filter_menu_search(actions, text))

        menu.exec_(self._filter_buttons[col].mapToGlobal(self._filter_buttons[col].rect().bottomLeft()))

        selected = set()
        all_checked = True
        for action in actions:
            if action.isChecked():
                selected.add(action.text())
            else:
                all_checked = False

        if all_checked:
            self._column_filters.pop(col, None)
        else:
            self._column_filters[col] = selected

        self._apply_filters()
        self._update_filter_button_style(col)

    def _toggle_all_filter(self, actions):
        all_checked = all(action.isChecked() for action in actions)
        for action in actions:
            action.setChecked(not all_checked)

    def _filter_menu_search(self, actions, text):
        lower = text.lower()
        for action in actions:
            action.setVisible(lower in action.text().lower())

    def _update_filter_button_style(self, col):
        btn = self._filter_buttons.get(col)
        if not btn:
            return
        btn.set_active(col in self._column_filters)

    def _apply_filters(self):
        self.trade_table.setSortingEnabled(False)
        for row in range(self.trade_table.rowCount()):
            show = True
            for col, allowed in self._column_filters.items():
                item = self.trade_table.item(row, col)
                if item and item.text() not in allowed:
                    show = False
                    break
            self.trade_table.setRowHidden(row, not show)
        self.trade_table.setSortingEnabled(True)

    def _toggle_instrument_visibility(self):
        self._instrument_hidden = not self._instrument_hidden

        if self._instrument_hidden:
            header = self.trade_table.horizontalHeader()
            self._saved_sort_column = header.sortIndicatorSection()
            self._saved_sort_order = header.sortIndicatorOrder()

            self.trade_table.setSortingEnabled(False)

            self._original_instruments.clear()

            for row in range(self.trade_table.rowCount()):
                item = self.trade_table.item(row, 2)
                if item:
                    self._original_instruments.append((row, item.text()))
                    item.setText('*' * len(item.text()))

            self.instrument_toggle_button.setText("显示标的")
        else:
            for saved_row, original_text in self._original_instruments:
                item = self.trade_table.item(saved_row, 2)
                if item:
                    item.setText(original_text)

            self._original_instruments.clear()

            self.trade_table.setSortingEnabled(True)
            if self._saved_sort_column >= 0:
                self.trade_table.sortByColumn(self._saved_sort_column, self._saved_sort_order)

            self.instrument_toggle_button.setText("隐藏标的")

    def _populate_trade_table(self, trades_data):
        self.trade_table.setSortingEnabled(False)
        self.trade_table.setRowCount(len(trades_data))
        self._instrument_hidden = False
        self._original_instruments.clear()
        if hasattr(self, 'instrument_toggle_button'):
            self.instrument_toggle_button.setText("隐藏标的")
        for row, trade in enumerate(trades_data):
            action, color = trade["action"], QColor("#666666")
            if "买" in action:
                color, display_action = QColor("#ff4444"), "▲买入"
            elif "卖" in action:
                color, display_action = QColor("#44aa44"), "▼卖出"
            else:
                display_action = action

            price_text, pnl_text, fee_text, turnover_text = "--", "--", "--", "--"
            if trade["order_id"] != -1:
                price_text, fee_text = f"{trade['price']:.3f}", f"{trade['fee']:,.2f}"
                turnover_text = f"{trade['turnover']:,.2f}"
                if trade["pnl"] != 0:
                    pnl_text = f"{trade['pnl']:,.2f}"

            items_to_add = [
                str(trade["order_id"]),
                trade["time"].strftime("%Y-%m-%d %H:%M:%S"),
                trade["instrument"],
                display_action,
                str(trade["volume"]),
                f"{trade['order_price']:.3f}",
                price_text,
                turnover_text,
                pnl_text,
                fee_text,
                trade["memo"],
            ]
            for col, item_text in enumerate(items_to_add):
                item = QTableWidgetItem(item_text)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 3:
                    item.setForeground(color)
                if col == 8 and isinstance(trade["pnl"], (int, float)):
                    if trade["pnl"] > 0:
                        item.setForeground(QColor("#d14545"))
                    elif trade["pnl"] < 0:
                        item.setForeground(QColor("#3f993f"))
                self.trade_table.setItem(row, col, item)
        self.trade_table.setSortingEnabled(True)
        self._auto_adjust_all_columns()
        self._update_filter_button_positions()
        self._apply_filters()

    def _create_benchmark_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 20, 10, 10)
        df = self.plot_df
        if df is None or df.empty:
            layout.addWidget(
                QLabel(
                    "无有效交易日数据",
                    alignment=Qt.AlignCenter,
                    styleSheet="QLabel { font-size: 28px; font-weight: bold; }",
                )
            )
            return tab

        benchmark_df = self.result.benchmark_df
        if self.result.trade_start_date and benchmark_df is not None and not benchmark_df.empty:
            trade_start_dt = pd.to_datetime(self.result.trade_start_date)
            benchmark_df = benchmark_df[benchmark_df.index >= trade_start_dt].copy()
        benchmark_symbol = self.result.benchmark_symbol or "000300.SH"

        benchmark_display_name = self.BENCHMARK_NAME_MAP.get(benchmark_symbol, benchmark_symbol)

        if benchmark_df is None or benchmark_df.empty:
            layout.addWidget(
                QLabel(
                    f"无基准（{benchmark_display_name}）数据，请检查QMT数据是否同步",
                    alignment=Qt.AlignCenter,
                    styleSheet="QLabel { font-size: 18px; font-weight: bold; color: #d14545; }",
                )
            )
            return tab

        strategy_dates = df["datetime"].tolist()
        benchmark_dates = benchmark_df.index.tolist()

        benchmark_close_map = {}
        for dt in benchmark_dates:
            if hasattr(dt, "date"):
                benchmark_close_map[dt.date()] = benchmark_df.loc[dt, "close"]
            else:
                benchmark_close_map[pd.Timestamp(dt).date()] = benchmark_df.loc[dt, "close"]

        aligned_benchmark = []
        for dt in strategy_dates:
            if hasattr(dt, "date"):
                trade_date = dt.date()
            else:
                trade_date = pd.Timestamp(dt).date()
            if trade_date in benchmark_close_map:
                aligned_benchmark.append(benchmark_close_map[trade_date])
            else:
                aligned_benchmark.append(None)

        first_valid_idx = None
        for i, v in enumerate(aligned_benchmark):
            if v is not None:
                first_valid_idx = i
                break

        if first_valid_idx is None:
            layout.addWidget(
                QLabel(
                    f"策略交易日期范围内无基准（{benchmark_display_name}）数据",
                    alignment=Qt.AlignCenter,
                    styleSheet="QLabel { font-size: 18px; font-weight: bold; color: #d14545; }",
                )
            )
            return tab

        for i in range(first_valid_idx):
            aligned_benchmark[i] = aligned_benchmark[first_valid_idx]
        for i in range(first_valid_idx + 1, len(aligned_benchmark)):
            if aligned_benchmark[i] is None:
                aligned_benchmark[i] = aligned_benchmark[i - 1]

        date_strings = df["datetime"].dt.strftime("%Y-%m-%d").tolist()
        axis = DateAxis(x_values=date_strings, orientation="bottom")
        if 0 < len(date_strings) <= 3:
            axis.setTicks([[(i, date) for i, date in enumerate(date_strings)]])

        plot_widget = pg.PlotWidget(axisItems={"bottom": axis})
        plot_widget.setTitle("基准对比", color="k", size="16pt", bold=True)
        plot_widget.setBackground("#f7f7f7")
        plot_widget.showGrid(x=True, y=True, alpha=0.3)
        plot_widget.getPlotItem().layout.setContentsMargins(10, 25, 10, 10)
        view_box = plot_widget.getViewBox()
        view_box.setMouseEnabled(x=True, y=False)

        x = list(range(len(df.index)))
        strategy_equity = df["PortfolioValue"].tolist()
        initial_equity = strategy_equity[0]

        strategy_net_value = [eq / initial_equity for eq in strategy_equity]

        benchmark_initial = aligned_benchmark[0]
        benchmark_net_value = [b / benchmark_initial for b in aligned_benchmark]

        excess_return = [s / b for s, b in zip(strategy_net_value, benchmark_net_value)]

        strategy_curve = plot_widget.plot(
            x,
            strategy_net_value,
            pen=pg.mkPen("#dc3545", width=2.5, style=Qt.SolidLine),
            name="策略净值",
        )
        benchmark_curve = plot_widget.plot(
            x,
            benchmark_net_value,
            pen=pg.mkPen("#007bff", width=2.0, style=Qt.SolidLine),
            name=benchmark_display_name,
        )
        excess_curve = plot_widget.plot(
            x,
            excess_return,
            pen=pg.mkPen("#ff7f0e", width=1.5, style=Qt.SolidLine),
            name="超额收益",
        )

        legend = ClickableLegend()
        legend.setParentItem(plot_widget.getPlotItem().getViewBox())
        legend.setBrush(QBrush(QColor("#f0f0f0")))
        legend.setPen(QPen(QColor("#999999"), 1))
        legend.setLabelTextColor(QColor("#333333"))
        legend.setLabelTextSize("10pt")
        legend.setOffset((10, 10))
        plot_widget.getPlotItem().legend = legend
        legend.addItem(strategy_curve, "策略净值")
        legend.addItem(benchmark_curve, benchmark_display_name)
        legend.addItem(excess_curve, "超额收益")
        plot_widget.setLabel("left", "净值 / 超额收益", **{"color": "k", "font-size": "12pt"})

        margin = 2
        initial_x_min = -margin
        initial_x_max = len(x) - 1 + margin if x else margin
        plot_widget.setXRange(initial_x_min, initial_x_max, padding=0)

        vLine, hLine, info_label = self._create_crosshair_items(plot_widget)

        def mouse_moved_benchmark(event):
            pos = event
            vb = plot_widget.getViewBox()
            if vb.sceneBoundingRect().contains(pos):
                mousePoint = vb.mapSceneToView(pos)
                index = int(round(mousePoint.x())) if not np.isnan(mousePoint.x()) else -1
                if 0 <= index < len(df.index):
                    vLine.show()
                    hLine.show()
                    vLine.setPos(index)
                    hLine.setPos(mousePoint.y())
                    date_str = date_strings[index]
                    strategy_nv = strategy_net_value[index]
                    benchmark_nv = benchmark_net_value[index]
                    excess = excess_return[index]
                    info_label.setPlainText(
                        f"日期: {date_str}\n"
                        f"策略净值: {strategy_nv:.4f}\n"
                        f"{benchmark_display_name}净值: {benchmark_nv:.4f}\n"
                        f"超额收益: {excess:.4f}"
                    )
                    info_label.adjustSize()
                    self._update_tooltip_position(info_label, pos, vb)
                    info_label.show()
                else:
                    info_label.hide()
                    vLine.hide()
                    hLine.hide()
            else:
                info_label.hide()
                vLine.hide()
                hLine.hide()

        plot_widget.scene().sigMouseMoved.connect(mouse_moved_benchmark)
        layout.addWidget(plot_widget)
        return tab

    def _update_trade_table(self):
        if not hasattr(self, "all_processed_trades"):
            return
        filtered_trades = (
            self.all_processed_trades
            if self.show_untraded_checkbox.isChecked()
            else [t for t in self.all_processed_trades if t["order_id"] != -1]
        )
        self._populate_trade_table(filtered_trades)


def generate_report(result: "BacktestingResult"):
    app = QApplication.instance() or QApplication(sys.argv)
    window = BacktestReportWindow(result)
    window.show()
    app.exec_()
