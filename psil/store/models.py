from dataclasses import dataclass, field
import json


@dataclass
class Paper:
    doi: str
    title: str
    abstract: str = ""
    journal: str = ""
    authors: list[str] = field(default_factory=list)
    affiliations: list[str] = field(default_factory=list)
    pub_date: str = ""
    toc_image_url: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "Paper":
        authors = data.get("authors", [])
        if isinstance(authors, str):
            authors = json.loads(authors)
        affiliations = data.get("affiliations", [])
        if isinstance(affiliations, str):
            affiliations = json.loads(affiliations)
        return cls(
            doi=data["doi"],
            title=data["title"],
            abstract=data.get("abstract", ""),
            journal=data.get("journal", ""),
            authors=authors,
            affiliations=affiliations,
            pub_date=data.get("pub_date", ""),
            toc_image_url=data.get("toc_image_url", ""),
        )

    def to_dict(self) -> dict:
        return {
            "doi": self.doi,
            "title": self.title,
            "abstract": self.abstract,
            "journal": self.journal,
            "authors": self.authors,
            "affiliations": self.affiliations,
            "pub_date": self.pub_date,
            "toc_image_url": self.toc_image_url,
        }

    def to_db_row(self) -> dict:
        return {
            "doi": self.doi,
            "title": self.title,
            "abstract": self.abstract,
            "journal": self.journal,
            "authors": json.dumps(self.authors),
            "affiliations": json.dumps(self.affiliations),
            "pub_date": self.pub_date,
            "toc_image_url": self.toc_image_url,
        }
