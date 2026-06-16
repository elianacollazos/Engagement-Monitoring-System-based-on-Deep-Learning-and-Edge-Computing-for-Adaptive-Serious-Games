from .types import File, Report, Session


class ReportTemplate:
    name = "fer2013-engagement-report"


class ReportExporter:
    def export(self, report: Report, format: str) -> File:
        return File(path=f"{report.session.sessionId}.{format.lower()}")


class ReportGenerator:
    def __init__(
        self,
        reportTemplate: ReportTemplate | None = None,
        exporter: ReportExporter | None = None,
    ):
        self.reportTemplate = reportTemplate or ReportTemplate()
        self.exporter = exporter or ReportExporter()

    def generate(self, session: Session) -> Report:
        return Report(session=session, records=[])

    def export(self, report: Report, format: str) -> File:
        return self.exporter.export(report, format)
