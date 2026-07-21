from pptx import Presentation
from pptx.util import Inches
from pathlib import Path
from langchain_core.utils.uuid import uuid7

output_file = f"""./data/uploads/{uuid7()}.pptx"""

# Sample data for testing
slides = [
    {
        "slide_type": "summary",
        "title": "Executive Summary",
        "bullets": [
            "Revenue increased by 12%",
            "South region performed best",
            "Customer retention improved"
        ]
    },
    {
        "slide_type": "table",
        "title": "Revenue by Region",
        "table": {
            "columns": [
                "Region",
                "Revenue",
                "Growth %"
            ],
            "rows": [
                ["South", "14000", "12%"],
                ["North", "13600", "10%"],
                ["West", "12250", "8%"],
                ["East", "11900", "6%"]
            ]
        }
    },
    {
        "slide_type": "chart",
        "title": "Revenue Trend",
        "charts": [
            "./data/019f7fff-4007-70d1-acbd-32dbce5bf11a.png"
        ]
    },
    {
        "slide_type": "image",
        "title": "Architecture Diagram",
        "images": [
            "./data/019f7fff-4026-7773-ae5b-a6d871386170.png"
        ]
    }
]

# Function that creates ppt from the content
def create_presentation(slides):
    """
    Create a PowerPoint presentation from a standardized slide schema.

    Supported slide types:
    - bullet
    - summary
    - table
    - chart
    - image
    """

    prs = Presentation()

    for slide_data in slides:

        slide_type = slide_data.get(
            "slide_type",
            "bullet"
        ).lower()

        # Use Title and Content layout
        slide = prs.slides.add_slide(
            prs.slide_layouts[1]
        )

        # -------------------------
        # Title
        # -------------------------
        slide.shapes.title.text = slide_data.get(
            "title",
            ""
        )

        # -------------------------
        # Subtitle / Description
        # -------------------------
        subtitle = slide_data.get(
            "subtitle",
            ""
        )

        description = slide_data.get(
            "description",
            ""
        )

        content_text = []

        if subtitle:
            content_text.append(subtitle)

        if description:
            content_text.append(description)

        # ----------------------------------
        # BULLET / SUMMARY SLIDES
        # ----------------------------------
        if slide_type in ["bullet", "summary"]:

            bullets = slide_data.get(
                "bullets",
                []
            )

            body = slide.placeholders[1].text_frame

            if content_text:
                body.text = "\n".join(content_text)

            for bullet in bullets:

                p = body.add_paragraph()
                p.text = bullet
                p.level = 0

        # ----------------------------------
        # TABLE SLIDES
        # ----------------------------------
        elif slide_type == "table":

            table_data = slide_data.get(
                "table",
                {}
            )

            columns = table_data.get(
                "columns",
                []
            )

            rows = table_data.get(
                "rows",
                []
            )

            if not columns:
                continue

            row_count = len(rows) + 1
            col_count = len(columns)

            table_shape = slide.shapes.add_table(
                row_count,
                col_count,
                Inches(0.5),
                Inches(1.5),
                Inches(9),
                Inches(3)
            )

            table = table_shape.table

            # Headers
            for col_idx, column_name in enumerate(columns):
                table.cell(
                    0,
                    col_idx
                ).text = str(column_name)

            # Rows
            for row_idx, row_values in enumerate(
                rows,
                start=1
            ):

                for col_idx, value in enumerate(
                    row_values
                ):
                    table.cell(
                        row_idx,
                        col_idx
                    ).text = str(value)

        # ----------------------------------
        # IMAGE SLIDES
        # ----------------------------------
        elif slide_type == "image":

            images = slide_data.get(
                "images",
                []
            )

            left = 0.5

            for image_path in images:
                file_path = Path(image_path)
                if file_path.is_file():
                    slide.shapes.add_picture(
                        image_path,
                        Inches(left),
                        Inches(1.5),
                        width=Inches(4)
                    )

                left += 4.2

        # ----------------------------------
        # CHART SLIDES
        # ----------------------------------
        elif slide_type == "chart":

            charts = slide_data.get(
                "charts",
                []
            )

            left = 0.5

            for chart_path in charts:
                file_path = Path(chart_path)
                if file_path.is_file():
                    slide.shapes.add_picture(
                        chart_path,
                        Inches(left),
                        Inches(1.5),
                        width=Inches(4.5)
                    )

                left += 4.8

        # ----------------------------------
        # MIXED CONTENT
        # ----------------------------------
        else:

            bullets = slide_data.get(
                "bullets",
                []
            )

            if bullets:

                body = slide.placeholders[1].text_frame

                for bullet in bullets:
                    p = body.add_paragraph()
                    p.text = bullet

    prs.save(output_file)

    return output_file 
