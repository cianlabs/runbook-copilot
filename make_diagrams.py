"""Draw the agent-loop diagrams in images/.

Same idea as images/m1-recap.png and images/m2-recap.png: a light background,
rounded cards, short plain-language labels, and one idea per box. Run:

    python make_diagrams.py

No network, no API key. The pictures contain no secrets.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent / "images"
SCALE = 2  # draw large, then shrink, so text and curves stay smooth

BG = (246, 247, 251)
INK = (27, 35, 51)
MUTED = (70, 82, 107)
LABEL = (110, 122, 145)
ARROW = (90, 102, 128)
WHITE = (255, 255, 255)

BLUE = ((231, 245, 255), (51, 154, 240), (24, 100, 171))
PURPLE = ((243, 240, 255), (112, 72, 232), (88, 52, 190))
GREEN = ((235, 251, 238), (47, 158, 68), (27, 120, 55))
YELLOW = ((255, 249, 219), (240, 160, 0), (160, 100, 0))
ORANGE = ((255, 244, 224), (230, 119, 0), (168, 78, 0))
RED = ((255, 240, 240), (224, 49, 49), (176, 32, 32))
TEAL = ((232, 250, 247), (18, 148, 136), (12, 110, 102))

FONT_DIR = Path("/usr/share/fonts/truetype/macos")
FONT_FILE = {
    "regular": FONT_DIR / "Inter-Regular.ttf",
    "medium": FONT_DIR / "Inter-Medium.ttf",
    "semibold": FONT_DIR / "Inter-SemiBold.ttf",
    "bold": FONT_DIR / "Inter-Bold.ttf",
}


def font(size, weight="regular"):
    return ImageFont.truetype(str(FONT_FILE[weight]), size * SCALE)


class Canvas:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.image = Image.new("RGB", (width * SCALE, height * SCALE), BG)
        self.draw = ImageDraw.Draw(self.image)

    def _s(self, *points):
        return tuple(int(round(v * SCALE)) for v in points)

    def text(self, xy, text, size, fill, weight="regular"):
        self.draw.text(self._s(*xy), text, font=font(size, weight), fill=fill)

    def text_width(self, text, size, weight="regular"):
        return self.draw.textlength(text, font=font(size, weight)) / SCALE

    def wrap(self, text, size, max_width, weight="regular"):
        lines = []
        for paragraph in text.split("\n"):
            words = paragraph.split()
            if not words:
                lines.append("")
                continue
            current = words[0]
            for word in words[1:]:
                trial = current + " " + word
                if self.text_width(trial, size, weight) <= max_width:
                    current = trial
                else:
                    lines.append(current)
                    current = word
            lines.append(current)
        return lines

    def rounded(self, box, fill, outline, radius=16, width=2):
        x1, y1, x2, y2 = box
        self.draw.rounded_rectangle(
            self._s(x1, y1, x2, y2),
            radius=radius * SCALE,
            fill=fill,
            outline=outline,
            width=width * SCALE,
        )

    def line(self, start, end, fill=ARROW, width=3):
        self.draw.line([self._s(*start), self._s(*end)], fill=fill, width=width * SCALE)

    def arrow_right(self, x1, x2, y, fill=ARROW):
        self.line((x1, y), (x2 - 9, y), fill=fill)
        self.draw.polygon(
            [self._s(x2, y), self._s(x2 - 11, y - 6), self._s(x2 - 11, y + 6)],
            fill=fill,
        )

    def arrow_down(self, x, y1, y2, fill=ARROW):
        self.line((x, y1), (x, y2 - 9), fill=fill)
        self.draw.polygon(
            [self._s(x, y2), self._s(x - 6, y2 - 11), self._s(x + 6, y2 - 11)],
            fill=fill,
        )

    def arrow_up(self, x, y_from, y_to, fill=ARROW):
        """Arrow pointing up. y_from is below y_to."""
        self.line((x, y_from), (x, y_to + 9), fill=fill)
        self.draw.polygon(
            [self._s(x, y_to), self._s(x - 6, y_to + 11), self._s(x + 6, y_to + 11)],
            fill=fill,
        )

    def arrow_left(self, x1, x2, y, fill=ARROW):
        """Arrow from x1 toward the smaller x2."""
        self.line((x1, y), (x2 + 9, y), fill=fill)
        self.draw.polygon(
            [self._s(x2, y), self._s(x2 + 11, y - 6), self._s(x2 + 11, y + 6)],
            fill=fill,
        )

    def label(self, xy, text):
        """Small tracked uppercase section label."""
        x, y = xy
        face = font(13, "semibold")
        for ch in text.upper():
            self.draw.text(self._s(x, y), ch, font=face, fill=LABEL)
            x += self.draw.textlength(ch, font=face) / SCALE + 1.15

    def card(self, box, fill, edge, title, body, title_fill=None, title_size=18, body_size=15):
        self.rounded(box, fill, edge)
        x1, y1, x2, y2 = box
        pad = 16
        title_fill = title_fill or INK
        self.text((x1 + pad, y1 + 14), title, title_size, title_fill, "bold")
        lines = self.wrap(body, body_size, (x2 - x1) - pad * 2, "regular")
        y = y1 + 14 + title_size + 10
        for line in lines:
            self.text((x1 + pad, y), line, body_size, MUTED, "regular")
            y += body_size + 6
        return y

    def save(self, name):
        OUT_DIR.mkdir(exist_ok=True)
        path = OUT_DIR / name
        out = self.image.resize((self.width, self.height), Image.Resampling.LANCZOS)
        out.save(path, "PNG")
        return path


def header(canvas, title, subtitle):
    canvas.text((48, 36), title, 36, INK, "bold")
    y = 88
    for line in canvas.wrap(subtitle, 17, canvas.width - 96):
        canvas.text((48, y), line, 17, MUTED)
        y += 26
    return y


def agent_loop():
    c = Canvas(1600, 1020)
    header(
        c,
        "Agent loop",
        "You ask a question. The assistant thinks, searches the runbooks, reads what comes back, then repeats or stops.",
    )
    c.label((48, 156), "The repeating cycle")

    gap = 22
    left = 48
    right = 1552
    n = 5
    box_w = (right - left - gap * (n - 1)) / n
    y1, h = 192, 168
    steps = [
        (BLUE, "1. You ask", "Your question, in ordinary words."),
        (PURPLE, "2. Think", "What do I still need to know from the runbooks?"),
        (GREEN, "3. search_runbooks", "The tool looks up sections. It only reads. It changes nothing."),
        (YELLOW, "4. Sections read", "Matches are printed and remembered, once per section."),
        (ORANGE, "5. Repeat or stop", "Search again, or stop for one of the four reasons below."),
    ]
    boxes = []
    x = left
    for colors, title, body in steps:
        box = (x, y1, x + box_w, y1 + h)
        c.card(box, colors[0], colors[1], title, body, title_size=16, body_size=14)
        boxes.append(box)
        x += box_w + gap
    mid_y = y1 + h / 2
    for i in range(n - 1):
        c.arrow_right(boxes[i][2] + 3, boxes[i + 1][0] - 3, mid_y)

    # Repeat goes back to Think, under the row.
    back_y = y1 + h + 28
    think = boxes[1]
    last = boxes[4]
    start_x = (last[0] + last[2]) / 2
    end_x = (think[0] + think[2]) / 2
    c.line((start_x, y1 + h), (start_x, back_y))
    c.line((start_x, back_y), (end_x, back_y))
    c.arrow_up(end_x, back_y, y1 + h)
    label = "Repeat"
    label_w = c.text_width(label, 14, "semibold")
    chip_x = (start_x + end_x) / 2 - label_w / 2
    c.rounded((chip_x - 8, back_y - 24, chip_x + label_w + 8, back_y - 4), BG, BG, radius=6, width=1)
    c.text((chip_x, back_y - 22), label, 14, ORANGE[2], "semibold")

    c.label((48, 452), "Four stop reasons")
    reasons = [
        (GREEN, "ANSWERED", "The model wrote the answer. This is the normal stop."),
        (RED, "NO_RELEVANT_RUNBOOK", "The first search found nothing. The reply is \"I don't know\". No Sources list."),
        (ORANGE, "MAX_STEPS_REACHED", "All 4 steps were used and there is still no answer. The loop stops."),
        (PURPLE, "ACTION_REQUESTED", "The model asked for execute_action. The approval gate runs next."),
    ]
    ry, rh = 488, 176
    rw = (right - left - 18 * 3) / 4
    x = left
    for colors, title, body in reasons:
        c.card((x, ry, x + rw, ry + rh), colors[0], colors[1], title, body, title_fill=colors[2], title_size=15, body_size=14)
        x += rw + 18

    c.label((48, 700), "Where the Sources list is built")
    panel = (48, 736, 1552, 960)
    c.rounded(panel, WHITE, (201, 209, 228))
    c.text((72, 756), "Built here, when the loop stops", 20, INK, "bold")
    bullets = [
        "Only sections that search_runbooks actually returned can be cited. A title that was never read is never listed.",
        "A section is cited when its best score is 0.15 or higher, or when the answer names that section's title.",
        "A weak extra hit (for example High swap usage at 0.085) is left off, unless the answer uses it by name.",
        "If every score is under 0.15, the loop does not answer (WEAK_MATCH_ONLY) and Sources stays empty.",
    ]
    y = 800
    for bullet in bullets:
        c.rounded((76, y + 6, 88, y + 18), BLUE[1], BLUE[1], radius=6, width=1)
        lines = c.wrap(bullet, 16, 1420)
        for i, line in enumerate(lines):
            c.text((100, y), line, 16, MUTED)
            y += 24
        y += 10
    c.save("agent-loop.png")


def approval_gate():
    c = Canvas(1600, 1040)
    header(
        c,
        "Approval gate",
        "The model can ask for an action. Only a person can let it run, and the run is only a simulation.",
    )
    c.label((48, 148), "Two ways an action is proposed")

    top = [
        (BLUE, (48, 184, 760, 320), "Model asks for an action", "It calls execute_action. The loop stops with ACTION_REQUESTED, then the gate runs."),
        (ORANGE, (840, 184, 1552, 320), "--force-action skips the model", "The -f flag goes straight to this same gate with a sample restart. No API key is used."),
    ]
    for colors, box, title, body in top:
        c.card(box, colors[0], colors[1], title, body, title_size=18, body_size=15)

    # Both paths join the gate. The section label stays left of the arrows.
    c.arrow_down(404, 320, 378)
    c.arrow_down(1196, 320, 378)
    c.line((404, 378), (1196, 378))
    c.arrow_down(800, 378, 448)
    c.label((48, 400), "The human approval gate")
    c.card(
        (220, 460, 1380, 620),
        YELLOW[0],
        YELLOW[1],
        "Approve this action? (y/n)",
        "The gate shows the action and any related runbook sections. Only the words y or yes (any case) pass. The model cannot approve its own action.",
        title_size=22,
        body_size=16,
    )

    c.arrow_down(520, 620, 748)
    c.arrow_down(1080, 620, 748)
    c.text((390, 662), "y or yes", 15, GREEN[2], "bold")
    c.text((1110, 662), "anything else", 15, RED[2], "bold")

    c.label((48, 716), "What happens next")
    c.card(
        (48, 752, 760, 960),
        GREEN[0],
        GREEN[1],
        "Simulated action runs",
        "The loop prints realistic steps, such as stopping a process and starting it again. It returns a success or failure result. No real process is signaled.",
        title_fill=GREEN[2],
    )
    c.card(
        (840, 752, 1552, 960),
        RED[0],
        RED[1],
        "Denied. Nothing was run.",
        "n, a blank line, or any other word denies it. The simulation does not start. The log says the action was denied.",
        title_fill=RED[2],
    )
    c.save("approval-gate.png")


def stop_reasons():
    c = Canvas(1600, 1180)
    header(
        c,
        "Stop reasons",
        "Every finished question prints one Stop reason. Follow the branch that matches what the model just did.",
    )
    c.label((48, 148), "What did the model do on this step?")

    gap = 20
    left = 48
    col_w = (1552 - left - gap * 2) / 3
    xs = [left + i * (col_w + gap) for i in range(3)]

    # Columns 1 and 2 are a single choice. Column 3 keeps going, because an
    # empty search means two different things.
    branches = [
        (BLUE, xs[0], "No tool call", "It wrote a reply and stopped calling tools.", "ANSWERED", "Normal stop. Sources is added only for sections that were actually read."),
        (PURPLE, xs[1], "execute_action", "It asked to do something, not to look something up.", "ACTION_REQUESTED", "The approval gate runs next. Only y or yes lets the simulated action run."),
    ]
    for colors, x, title, body, outcome, detail in branches:
        c.card((x, 184, x + col_w, 320), colors[0], colors[1], title, body, title_size=18, body_size=15)
        c.arrow_down(x + col_w / 2, 320, 360)
        c.card((x, 368, x + col_w, 560), WHITE, colors[1], outcome, detail, title_fill=colors[2], title_size=16, body_size=15)

    x = xs[2]
    c.card((x, 184, x + col_w, 300), TEAL[0], TEAL[1], "search_runbooks", "It asked the runbooks a short question.", title_size=18, body_size=15)
    c.arrow_down(x + col_w / 2, 300, 336)
    c.card(
        (x, 344, x + col_w, 430),
        WHITE,
        TEAL[1],
        "Did that search find sections?",
        "Empty or not, and was anything already read?",
        title_fill=TEAL[2],
        title_size=15,
        body_size=14,
    )
    c.arrow_down(x + col_w / 2, 430, 468)
    c.text((x + 16, 436), "if the result is empty", 13, MUTED, "medium")
    c.card(
        (x, 476, x + col_w, 640),
        RED[0],
        RED[1],
        "First search is empty",
        "Nothing was read yet. Stop: NO_RELEVANT_RUNBOOK. The loop says \"I don't know\" and does not guess. No Sources list.",
        title_fill=RED[2],
        title_size=15,
        body_size=14,
    )
    c.text((x + 16, 648), "or, if sections were already read", 13, MUTED, "medium")
    c.card(
        (x, 672, x + col_w, 836),
        GREEN[0],
        GREEN[1],
        "A later search is empty",
        "Do not stop. Do not erase the sections already read. Let the model answer or search again.",
        title_fill=GREEN[2],
        title_size=15,
        body_size=14,
    )
    c.card(
        (x, 852, x + col_w, 980),
        TEAL[0],
        TEAL[1],
        "Sections came back",
        "Remember them and continue. That is not a stop by itself.",
        title_fill=TEAL[2],
        title_size=15,
        body_size=14,
    )

    c.label((48, 920), "One more stop, if the loop runs out of steps")
    c.card(
        (48, 956, xs[2] - 20, 1120),
        ORANGE[0],
        ORANGE[1],
        "MAX_STEPS_REACHED  ·  4 steps",
        "Each question may take 4 model steps. If every step still calls a tool, the loop stops instead of going forever. It cites what the citation rule kept and does not guess past the sections it already read.",
        title_fill=ORANGE[2],
        title_size=18,
        body_size=15,
    )
    c.save("stop-reasons.png")


def guardrails():
    """Week 4: I-don't-know stops, the hard step cap, and the local page."""
    c = Canvas(1600, 1180)
    header(
        c,
        "Guardrails and the page",
        "The copilot says it does not know rather than guess. It cannot take more than 4 steps. A person clicks before anything runs.",
    )

    c.label((48, 150), "Three ways it refuses to guess")
    cards = [
        (RED, "NO_RELEVANT_RUNBOOK", "Search returned nothing. The reply is \"I don't know\". No Sources list. The model is not asked to invent steps."),
        (ORANGE, "WEAK_MATCH_ONLY", "Every score was under 0.15, and nothing stronger was read earlier. Too weak to trust. No Sources list."),
        (PURPLE, "UNGROUNDED_ANSWER", "The model wrote steps before reading a section. Those words are thrown away. A short \"more detail\" question can still be shown."),
    ]
    left, right = 48, 1552
    gap = 18
    width = (right - left - gap * 2) / 3
    x = left
    for colors, title, body in cards:
        c.card((x, 186, x + width, 400), colors[0], colors[1], title, body, title_fill=colors[2], title_size=16, body_size=15)
        x += width + gap

    c.label((48, 432), "The hard step cap")
    c.card(
        (48, 468, 1552, 640),
        YELLOW[0],
        YELLOW[1],
        "MAX_STEPS = 4, checked before every model call",
        "The loop counts model turns. When the count is already 4, it does not call the model again. A model that would search forever still stops. The stop reason is MAX_STEPS_REACHED, and the message says it will not guess past the sections it already read.",
        title_fill=ORANGE[2],
        title_size=20,
        body_size=16,
    )

    c.label((48, 672), "The local page")
    page = [
        (BLUE, "1. Ask", "One command starts the page. Type the problem in ordinary words. No terminal needed."),
        (GREEN, "2. Read the result", "Numbered steps, the Sources list, the stop reason in plain words, and the step-by-step trace."),
        (YELLOW, "3. Click to approve", "If it proposes an action, nothing runs until Approve. Do not run refuses. The run is only a simulation."),
    ]
    box_w = (right - left - gap * 2) / 3
    x = left
    for colors, title, body in page:
        c.card((x, 708, x + box_w, 900), colors[0], colors[1], title, body, title_size=18, body_size=15)
        x += box_w + gap
    mid = 708 + 96
    # arrows between the three page cards
    x = left
    for i in range(2):
        c.arrow_right(x + box_w + 2, x + box_w + gap - 2, mid)
        x += box_w + gap

    c.label((48, 932), "What the page will not do")
    notes = [
        "No API key on screen. Practice mode runs with a built-in model when XAI_API_KEY is unset. Live mode reads the key from the terminal and does not print it.",
        "The Approve button is a separate click. Opening the page, or asking a question, never runs the action.",
        "A weak match can still be printed in the trace so you can see why it was refused. It is not a citation.",
    ]
    y = 968
    for note in notes:
        c.rounded((56, y + 6, 68, y + 18), TEAL[1], TEAL[1], radius=6, width=1)
        lines = c.wrap(note, 16, 1460)
        for line in lines:
            c.text((80, y), line, 16, MUTED)
            y += 24
        y += 8
    c.save("guardrails.png")


def main():
    agent_loop()
    approval_gate()
    stop_reasons()
    guardrails()
    print("Wrote images/agent-loop.png, images/approval-gate.png, images/stop-reasons.png, images/guardrails.png")


if __name__ == "__main__":
    main()
