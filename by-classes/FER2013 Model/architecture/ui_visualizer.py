from .temporal_record import TemporalRecord


class DashboardView:
    def render(self, record: TemporalRecord):
        return record.toJSON()


class UIVisualizer:
    def __init__(self, dashboard: DashboardView | None = None):
        self.dashboard = dashboard or DashboardView()

    def update(self, record: TemporalRecord) -> None:
        self.dashboard.render(record)

    def showIndicators(self, indicators) -> None:
        return None

    def showAlerts(self, alerts) -> None:
        return None
