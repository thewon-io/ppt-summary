import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageOps, ImageEnhance, ImageDraw
import io
import os
import re
import json
import cv2
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_ST_COORDS = True
except ImportError:
    HAS_ST_COORDS = False

# ---------------------------------------------------------
# 1. 한글 폰트 및 디자인 테마 설정
# ---------------------------------------------------------
FONT_NAME = "Helvetica"
if os.path.exists("malgun.ttf"):
    try:
        pdfmetrics.registerFont(TTFont("Malgun", "malgun.ttf"))
        FONT_NAME = "Malgun"
    except Exception:
        pass

COLOR_PRIMARY = colors.HexColor('#0F172A')
COLOR_SECONDARY = colors.HexColor('#0284C7')
COLOR_BG_LIGHT = colors.HexColor('#F8FAFC')
COLOR_TEXT = colors.HexColor('#334155')

# ---------------------------------------------------------
# 2. PDF 쪽수 자동 표기 커스텀 캔버스
# ---------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.setFont(FONT_NAME, 8)
        self.setFillColor(COLOR_TEXT)
        page_text = f"{self._pageNumber} / {page_count}"
        self.drawCentredString(A4[0] / 2.0, 20, page_text)

# ---------------------------------------------------------
# 3. 이미지 투시 변환 및 4점 조정 함수
# ---------------------------------------------------------
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_perspective_4pts(image_pil, pts):
    img_np = np.array(image_pil)
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    rect = order_points(np.array(pts, dtype="float32"))
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB), 200)

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB), 150)

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped_np = cv2.warpPerspective(img_np, M, (maxWidth, maxHeight))
    return Image.fromarray(warped_np)

def detect_slide_contour(image_pil):
    image_pil = ImageOps.exif_transpose(image_pil).convert('RGB')
    w, h = image_pil.size
    default_pts = np.array([[w*0.05, h*0.05], [w*0.95, h*0.05], [w*0.95, h*0.95], [w*0.05, h*0.95]], dtype="float32")

    try:
        img_np = np.array(image_pil)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        ratio = h / 600.0
        small = cv2.resize(img_cv, (int(w / ratio), 600))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours[:3]:
                if cv2.contourArea(cnt) > 0.1 * (600 * (w / ratio)):
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        return approx.reshape(4, 2) * ratio
    except Exception:
        pass
    return default_pts

def draw_guide_overlay(image_pil, pts):
    """디스플레이 화면용 오버레이 레이어 생성"""
    overlay = image_pil.copy()
    draw = ImageDraw.Draw(overlay)
    rect = order_points(np.array(pts, dtype="float32"))
    
    polygon = [tuple(p) for p in rect]
    draw.polygon(polygon, outline="#EF4444", width=3)
    
    r = 8  # 가독성 확보를 위한 선명한 고정 점 크기
    for p in rect:
        draw.ellipse((p[0]-r, p[1]-r, p[0]+r, p[1]+r), fill="#EF4444", outline="#FFFFFF", width=2)
        
    return overlay

# ---------------------------------------------------------
# 4. PDF 보고서 생성 함수
# ---------------------------------------------------------
def create_pdf_summary(data_dict, slide_images):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('MainTitle', parent=styles['Heading1'], fontName=FONT_NAME, fontSize=20, leading=26, textColor=COLOR_PRIMARY, alignment=1)
    sub_heading_style = ParagraphStyle('SubHeading', parent=styles['Heading2'], fontName=FONT_NAME, fontSize=12, leading=16, textColor=colors.white)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontName=FONT_NAME, fontSize=8.5, leading=13, textColor=COLOR_TEXT, wordWrap='CJK')
    bold_body_style = ParagraphStyle('BoldBody', parent=body_style, fontName=FONT_NAME)
    slide_hdr_style = ParagraphStyle('SlideHdr', parent=styles['Normal'], fontName=FONT_NAME, fontSize=10, leading=14, textColor=colors.white, alignment=1)

    story = []

    def make_header_box(title_text):
        t = Table([[Paragraph(f"<b>{title_text}</b>", sub_heading_style)]], colWidths=[520])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), COLOR_PRIMARY),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
        ]))
        return t

    story.append(Paragraph("<b>PPT SUMMARY</b>", title_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=COLOR_SECONDARY, spaceAfter=12))

    story.append(make_header_box("배경 정보"))
    story.append(Spacer(1, 6))
    
    bg_info = data_dict.get("bg_info", {})
    bg_data = [
        [Paragraph("<b>구분</b>", bold_body_style), Paragraph("<b>세부 내용</b>", bold_body_style)],
        [Paragraph("<b>일시</b>", body_style), Paragraph(bg_info.get("date", "-"), body_style)],
        [Paragraph("<b>주최/장소</b>", body_style), Paragraph(bg_info.get("place", "-"), body_style)],
        [Paragraph("<b>발표자</b>", body_style), Paragraph(bg_info.get("speaker", "-"), body_style)],
        [Paragraph("<b>발표 제목</b>", body_style), Paragraph(bg_info.get("title", "-"), body_style)],
        [Paragraph("<b>키워드</b>", body_style), Paragraph(bg_info.get("keywords", "-"), body_style)],
        [Paragraph("<b>한줄 요약</b>", body_style), Paragraph(bg_info.get("why", "-"), body_style)],
    ]
    bg_table = Table(bg_data, colWidths=[120, 400])
    bg_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), COLOR_SECONDARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('BACKGROUND', (0,1), (0,-1), COLOR_BG_LIGHT),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(bg_table)
    story.append(Spacer(1, 12))

    story.append(make_header_box("핵심 요약"))
    story.append(Spacer(1, 6))
    
    summary_text = data_dict.get("executive_summary", "핵심 요약 내용이 없습니다.")
    sum_table = Table([[Paragraph(summary_text.replace("\n", "<br/>"), body_style)]], colWidths=[520])
    sum_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), COLOR_BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 0.8, COLOR_SECONDARY),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(sum_table)
    story.append(Spacer(1, 14))

    story.append(make_header_box("자료 정리"))
    story.append(Spacer(1, 10))

    slides_data = data_dict.get("slides", [])
    for idx, sdata in enumerate(slides_data):
        slide_tbl_data = []
        slide_tbl_data.append([Paragraph(f"<b>슬라이드 #{idx+1}</b>", slide_hdr_style), ""])
        
        if idx < len(slide_images):
            img_buf = io.BytesIO()
            slide_images[idx].save(img_buf, format='JPEG', quality=90)
            img_buf.seek(0)
            from reportlab.platypus import Image as RLImage
            rl_img = RLImage(img_buf, width=320, height=180)
            slide_tbl_data.append([rl_img, ""])
        else:
            slide_tbl_data.append([Paragraph("이미지 없음", body_style), ""])

        slide_tbl_data.append([Paragraph("<b>제목</b>", body_style), Paragraph(sdata.get("title", ""), body_style)])
        slide_tbl_data.append([Paragraph("<b>내용</b>", body_style), Paragraph(sdata.get("content", "").replace("\n", "<br/>"), body_style)])
        slide_tbl_data.append([Paragraph("<b>요점</b>", body_style), Paragraph(sdata.get("keypoint", ""), body_style)])

        if sdata.get("audio"):
            slide_tbl_data.append([
                Paragraph("<b>녹음 자료</b>", body_style),
                Paragraph(f"<b>[녹음 #{idx+1}]</b><br/>{sdata['audio']}", body_style)
            ])

        if sdata.get("memo"):
            slide_tbl_data.append([
                Paragraph("<b>메모 자료</b>", body_style),
                Paragraph(f"<b>[메모 #{idx+1}]</b><br/>{sdata['memo']}", body_style)
            ])

        s_table = Table(slide_tbl_data, colWidths=[80, 440])
        
        t_style = [
            ('SPAN', (0,0), (1,0)),
            ('SPAN', (0,1), (1,1)),
            ('BACKGROUND', (0,0), (1,0), COLOR_PRIMARY),
            ('ALIGN', (0,0), (1,0), 'CENTER'),
            ('ALIGN', (0,1), (1,1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('BACKGROUND', (0,2), (0,-1), COLOR_BG_LIGHT),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]
        s_table.setStyle(TableStyle(t_style))
        story.append(s_table)
        story.append(Spacer(1, 14))

    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer

# ---------------------------------------------------------
# 5. TXT/Markdown 보고서 생성 함수
# ---------------------------------------------------------
def create_txt_summary(data_dict):
    bg = data_dict.get("bg_info", {})
    summary = data_dict.get("executive_summary", "")
    slides = data_dict.get("slides", [])

    txt = []
    txt.append("# PPT SUMMARY\n")
    txt.append("---")
    
    txt.append("## 1. 배경 정보\n")
    txt.append("| 구분 | 내용 |")
    txt.append("| --- | --- |")
    txt.append(f"| 언제 (일시) | {bg.get('date', '-')} |")
    txt.append(f"| 어디서 (행사/장소) | {bg.get('place', '-')} |")
    txt.append(f"| 누가 (발표자) | {bg.get('speaker', '-')} |")
    txt.append(f"| 무엇을 (발표 제목) | {bg.get('title', '-')} |")
    txt.append(f"| 어떻게 (강연 키워드) | {bg.get('keywords', '-')} |")
    txt.append(f"| 왜 (요점 한줄 요약) | {bg.get('why', '-')} |\n")

    txt.append("## 2. 핵심 요약\n")
    txt.append(f"{summary}\n")

    txt.append("## 3. 자료 정리\n")
    for idx, s in enumerate(slides):
        txt.append(f"### 슬라이드 #{idx+1}\n")
        txt.append("| 구분 | 상세 내용 |")
        txt.append("| --- | --- |")
        txt.append(f"| 제목 | {s.get('title', '')} |")
        
        content_text = s.get('content', '').replace('\n', ' ')
        txt.append(f"| 내용 | {content_text} |")
        txt.append(f"| 요점 | {s.get('keypoint', '')} |\n")

        if s.get("audio"):
            txt.append(f"> 🎙️ **[녹음 #{idx+1}]**\n> {s['audio']}\n")
        if s.get("memo"):
            txt.append(f"> 📝 **[메모 #{idx+1}]**\n> {s['memo']}\n")
        txt.append("\n---")

    return "\n".join(txt)

# ---------------------------------------------------------
# 6. Streamlit 메인 앱 UI
# ---------------------------------------------------------
st.set_page_config(page_title="PPT SUMMARY 생성기", layout="wide")

st.title("📚 SlideReport AI (PPT 정밀 분석 및 SUMMARY 보고서 생성기)")
st.caption("마우스 클릭 4점 영역 조정 | 이미지 자동 추출 배경정보 | PDF & TXT 다운로드")

if "slides_store" not in st.session_state:
    st.session_state.slides_store = []

with st.sidebar:
    st.header("🔑 API 및 메타 설정")
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.subheader("📋 배경 정보 (미입력 시 AI가 자동으로 추출합니다)")
    meta_date = st.text_input("일시", value="")
    meta_place = st.text_input("주최/장소", value="")
    meta_speaker = st.text_input("발표자", value="")
    meta_title = st.text_input("발표 제목", value="")

uploaded_files = st.file_uploader("슬라이드 이미지 파일(PNG, JPG)을 선택하세요", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    file_names = [f.name for f in uploaded_files]
    stored_names = [item['name'] for item in st.session_state.slides_store]

    if file_names != stored_names:
        new_store = []
        with st.spinner("⚡ 모든 슬라이드의 영역을 자동 감지하고 평면 보정을 적용 중입니다..."):
            for f in uploaded_files:
                orig_img = Image.open(f).convert('RGB')
                auto_pts = detect_slide_contour(orig_img)
                warped_img = warp_perspective_4pts(orig_img, auto_pts)
                
                new_store.append({
                    'name': f.name,
                    'orig': orig_img,
                    'pts': auto_pts,
                    'final': warped_img,
                    'audio': "",
                    'memo': ""
                })
        st.session_state.slides_store = new_store

if st.session_state.slides_store:
    st.divider()
    st.subheader(f"🖼️ 1차 보정 결과 (총 {len(st.session_state.slides_store)}장)")
    
    cols = st.columns(min(len(st.session_state.slides_store), 4))
    for idx, item in enumerate(st.session_state.slides_store):
        with cols[idx % 4]:
            st.image(item['final'], caption=f"슬라이드 #{idx+1}", use_container_width=True)

    st.divider()
    with st.expander("🖱️ 마우스 클릭으로 가이드선 영역 변경 및 메모 입력", expanded=True):
        if not HAS_ST_COORDS:
            st.warning("`pip install streamlit-image-coordinates` 패키지를 설치하면 마우스 클릭 조정이 가능합니다.")
        else:
            slide_idx = st.selectbox("수정할 슬라이드를 선택하세요", range(len(st.session_state.slides_store)), format_func=lambda x: f"슬라이드 #{x+1}")
            target_item = st.session_state.slides_store[slide_idx]
            orig_img = target_item['orig']
            pts = target_item['pts']

            col_img, col_memo = st.columns([1.2, 0.8])

            with col_img:
                st.markdown("**🎯 가이드 이미지 상의 빨간 점을 드래그해서 위치를 수정하세요**")
                
                if st.button("🔄 영역 좌표 초기화 (자동인식 상태로 복원)"):
                    st.session_state.slides_store[slide_idx]['pts'] = detect_slide_contour(orig_img)
                    st.session_state.slides_store[slide_idx]['final'] = warp_perspective_4pts(orig_img, st.session_state.slides_store[slide_idx]['pts'])
                    st.rerun()

                # 화면 표시용 이미지 축소 및 스케일링 계산
                max_disp_w = 600
                disp_scale = 1.0
                if orig_img.width > max_disp_w:
                    disp_scale = max_disp_w / float(orig_img.width)
                    disp_w = max_disp_w
                    disp_h = int(orig_img.height * disp_scale)
                    disp_img = orig_img.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
                else:
                    disp_img = orig_img.copy()

                disp_pts = pts * disp_scale

                guided_img = draw_guide_overlay(disp_img, disp_pts)
                coords = streamlit_image_coordinates(guided_img, key=f"coords_{slide_idx}")

                if coords is not None:
                    # 클릭 위치를 원본 좌표로 역산
                    cx = coords["x"] / disp_scale
                    cy = coords["y"] / disp_scale
                    last_click = st.session_state.get(f"last_click_{slide_idx}")
                    
                    if last_click != (cx, cy):
                        st.session_state[f"last_click_{slide_idx}"] = (cx, cy)
                        dists = [np.hypot(p[0] - cx, p[1] - cy) for p in pts]
                        closest_idx = np.argmin(dists)
                        
                        pts[closest_idx] = [cx, cy]
                        st.session_state.slides_store[slide_idx]['pts'] = pts
                        st.session_state.slides_store[slide_idx]['final'] = warp_perspective_4pts(orig_img, pts)
                        st.rerun()

            with col_memo:
                st.markdown("**👁️ 현재 지정 영역 최종 보정 결과**")
                st.image(target_item['final'], caption=f"슬라이드 #{slide_idx+1} 평면화 결과", use_container_width=True)
                
                st.markdown("---")
                st.markdown("**🎙️ / 📝 녹음 및 아이디어 메모**")
                st.session_state.slides_store[slide_idx]['audio'] = st.text_area("강연 녹음 내용 (녹음 #1)", value=target_item['audio'], placeholder="녹음된 내용을 텍스트로 기재...")
                st.session_state.slides_store[slide_idx]['memo'] = st.text_area("메모/그림 설명 (메모 #1)", value=target_item['memo'], placeholder="떠오른 아이디어나 메모 기재...")

    st.divider()
    if st.button("🚀 PPT Slide 분석 및 Summary report 생성", type="primary", use_container_width=True):
        if not api_key:
            st.error("Gemini API Key를 입력해주세요.")
        else:
            with st.spinner("🧠 AI가 슬라이드 이미지를 정밀 분석하여 PPT SUMMARY 데이터를 도출 중입니다..."):
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-3.6-flash")

                final_images = [item['final'] for item in st.session_state.slides_store]
                
                prompt = f"""
제공된 슬라이드 이미지들을 순서대로 분석하여 보고서 JSON 데이터를 생성하세요.

[요구사항]
1. bg_info (배경 정보):
   - date: "{meta_date}" (비어있다면 이미지에서 일시 추출/추론)
   - place: "{meta_place}" (비어있다면 이미지에서 행사명/장소 추출/추론)
   - speaker: "{meta_speaker}" (비어있다면 이미지에서 발표자 추출/추론)
   - title: "{meta_title}" (비어있다면 표지 이미지에서 발표 제목 추출)
   - keywords: 강연 내용의 핵심 키워드들 정리
   - why: 발표 주제 및 강조 요약 한줄 작성

2. executive_summary (핵심 요약):
   - 전체 발표 내용의 초록(Introduction, Background, Methods, Results, Discussion, Conclusion, 의의 등) 구조적 정리

3. slides (슬라이드별 내용):
   - title: 슬라이드 제목
   - content: 슬라이드 텍스트 및 그림 내용 추출하여 해석 정리
   - keypoint: 슬라이드 핵심 메시지

아래 JSON 포맷으로만 정확히 응답하세요:
{{
  "bg_info": {{
    "date": "...",
    "place": "...",
    "speaker": "...",
    "title": "...",
    "keywords": "...",
    "why": "..."
  }},
  "executive_summary": "...",
  "slides": [
    {{
      "title": "...",
      "content": "...",
      "keypoint": "..."
    }}
  ]
}}
"""
                try:
                    res = model.generate_content([prompt] + final_images)
                    json_match = re.search(r'\{.*\}', res.text, re.DOTALL)
                    if json_match:
                        ai_data = json.loads(json_match.group(0))
                    else:
                        ai_data = {
                            "bg_info": {"date": meta_date or "2026년", "place": meta_place or "온라인", "speaker": meta_speaker or "발표자", "title": meta_title or "슬라이드 발표", "keywords": "AI", "why": "핵심 주제 요약"},
                            "executive_summary": res.text,
                            "slides": [{"title": f"슬라이드 #{i+1}", "content": "내용 추출", "keypoint": "핵심 요약"} for i in range(len(final_images))]
                        }

                    for idx, sitem in enumerate(st.session_state.slides_store):
                        if idx < len(ai_data["slides"]):
                            ai_data["slides"][idx]["audio"] = sitem.get("audio", "")
                            ai_data["slides"][idx]["memo"] = sitem.get("memo", "")

                    st.session_state.summary_data = ai_data
                    st.success("✨ PPT SUMMARY 생성이 완료되었습니다!")

                except Exception as e:
                    st.error(f"분석 중 오류 발생: {e}")

    if "summary_data" in st.session_state:
        st.divider()
        st.subheader("📥 다운로드 양식 선택")

        final_images = [item['final'] for item in st.session_state.slides_store]
        pdf_bytes = create_pdf_summary(st.session_state.summary_data, final_images)
        txt_bytes = create_txt_summary(st.session_state.summary_data)

        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            st.download_button(
                label="📄 PPT SUMMARY.pdf 다운로드",
                data=pdf_bytes,
                file_name="PPT SUMMARY.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        with dl_col2:
            st.download_button(
                label="📝 PPT SUMMARY.txt 다운로드 (마크다운)",
                data=txt_bytes,
                file_name="PPT SUMMARY.txt",
                mime="text/plain",
                use_container_width=True
            )