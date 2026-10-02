from . import boards, medical, nhtsa, ntsb, reports_pdf

ADAPTERS = {"ntsb": ntsb, "boards": boards, "nhtsa": nhtsa, "medical": medical,
            "ntsb_pipeline": reports_pdf.make("ntsb_pipeline"), "ntsb_highway": reports_pdf.make("ntsb_highway"),
            "tsb_pipeline": reports_pdf.make("tsb_pipeline")}
PREFIX = {"ntsb": "ANTSB", "boards": "ABRD", "nhtsa": "ANHTSA", "medical": "AMED",
          "ntsb_pipeline": "ANTP", "ntsb_highway": "ANTH", "tsb_pipeline": "ATSP"}
