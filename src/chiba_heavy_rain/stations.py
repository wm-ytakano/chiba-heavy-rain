"""Current JMA precipitation stations in Chiba Prefecture."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Station:
    block_no: str
    name: str
    kind: str  # "s" for surface observatory, "a" for AMeDAS

    @property
    def annual_page(self) -> str:
        return f"annually_{self.kind}.php"

    @property
    def daily_page(self) -> str:
        return f"daily_{self.kind}1.php"


# The list is the set shown on JMA's Chiba station-selection page on 2026-08-16.
STATIONS = (
    Station("0375", "香取", "a"),
    Station("0376", "我孫子", "a"),
    Station("0377", "東庄", "a"),
    Station("0378", "成田", "a"),
    Station("47648", "銚子", "s"),
    Station("47682", "千葉", "s"),
    Station("0381", "茂原", "a"),
    Station("0382", "木更津", "a"),
    Station("0383", "鋸南", "a"),
    Station("0384", "鴨川", "a"),
    Station("47674", "勝浦", "s"),
    Station("47672", "館山", "s"),
    Station("0916", "佐倉", "a"),
    Station("1003", "横芝光", "a"),
    Station("1004", "大多喜", "a"),
    Station("1236", "船橋", "a"),
    Station("1238", "牛久", "a"),
    Station("1241", "坂畑", "a"),
)

