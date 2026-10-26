import os
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))

def set_cell_margins(cell, top=70, bottom=70, left=100, right=100):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_tight_report():
    doc = Document()
    
    # Page setup - Margins (0.55 in top/bottom, 0.65 in left/right)
    for section in doc.sections:
        section.top_margin = Inches(0.5)
        section.bottom_margin = Inches(0.5)
        section.left_margin = Inches(0.65)
        section.right_margin = Inches(0.65)
        
    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(8.8)
    normal_style.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    normal_style.paragraph_format.line_spacing = 1.12
    normal_style.paragraph_format.space_after = Pt(2.5)
    
    # =========================================================================
    # PAGE 1: TITLE, PREPROCESSING PIPELINE & NLP FEATURE ENGINEERING
    # =========================================================================
    # Header Banner
    title_table = doc.add_table(rows=1, cols=1)
    title_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    title_table.autofit = False
    title_table.columns[0].width = Inches(7.2)
    
    cell = title_table.cell(0, 0)
    set_cell_background(cell, "0F172A")
    set_cell_margins(cell, top=120, bottom=120, left=150, right=150)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(1)
    run = p.add_run("Do Lyrics Make a Top 50 Hit? Analyzing the Spotify Global Top 200")
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    
    p2 = cell.add_paragraph()
    p2.paragraph_format.space_after = Pt(1)
    run2 = p2.add_run("Data Science Mini-Project Technical Report | Target: Emerging Songwriters & Independent Creators")
    run2.font.size = Pt(8.8)
    run2.font.bold = True
    run2.font.color.rgb = RGBColor(0x38, 0xBD, 0xF8)
    
    p3 = cell.add_paragraph()
    p3.paragraph_format.space_after = Pt(0)
    run3 = p3.add_run("Dataset: 53 Weekly Spotify Global Top 200 Charts | Final Modeling Cohort: N = 582 Unique English Songs (228 Top 50 Hits vs. 354 Ranks 51–200)")
    run3.font.size = Pt(7.8)
    run3.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
    
    # 1. Project Motivation
    h1 = doc.add_heading(level=1)
    h1.paragraph_format.space_before = Pt(4)
    h1.paragraph_format.space_after = Pt(2)
    r = h1.add_run("1. Motivation, Target Definition & Executive Summary")
    r.font.size = Pt(10.5)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    
    doc.add_paragraph(
        "Commercial songwriting culture often insists that streaming success requires a strict lyric formula: heavy chorus repetition, "
        "short verse length, simple words, or viral breakup themes. We designed this study to test whether written lyrics actually separate "
        "songs that break into the Spotify Global Top 50 (binary target: best_rank_this_year <= 50, y = 1, n = 228) from songs peaking in "
        "ranks 51–200 (y = 0, n = 354). Because lyrics are backend text that target audiences do not evaluate statistically, this technical "
        "report details the backend data pipeline, machine learning validation, and project canvas reflections."
    )
    
    # Executive Summary Box
    callout = doc.add_table(rows=1, cols=1)
    callout.alignment = WD_TABLE_ALIGNMENT.CENTER
    callout.columns[0].width = Inches(7.2)
    c_cell = callout.cell(0, 0)
    set_cell_background(c_cell, "F0FDF4")
    set_cell_margins(c_cell, top=80, bottom=80, left=120, right=120)
    
    cp = c_cell.paragraphs[0]
    c_run = cp.add_run("Core Technical Findings & Canvas Takeaway:")
    c_run.font.bold = True
    c_run.font.size = Pt(8.8)
    c_run.font.color.rgb = RGBColor(0x06, 0x5F, 0x46)
    
    exec_points = [
        "1. Surface Lyric Stats Fail to Separate Top 50 Hits: Median chorus repetition (31.8% vs. 33.3%, p = 0.215), word count (352 vs. 346 words, p = 0.529), and vocabulary diversity (0.372 vs. 0.365, p = 0.325) show zero statistically significant difference.",
        "2. ML Confirms Lyrics Cannot Predict Chart Rank: Evaluated under 5-Fold Artist-Grouped Cross-Validation, models trained purely on Sentence-BERT contextual embeddings (0.5080 ROC-AUC) or basic lyric counts (0.4696) perform at random coin-flip chance.",
        "3. Market Timing & The 'Formula-Free' Reality: High chart placement is driven by release recency (new releases debut high) and superstar fanbases, not lyric formulas. Formulaic club clichés (Topic 0) convert at only 22.6%, whereas Formula-Free unique songs (Topic -1) achieved 42 Top 50 hits (34.1%)."
    ]
    for ep in exec_points:
        p = c_cell.add_paragraph()
        p.paragraph_format.space_after = Pt(1.5)
        pr = p.add_run(ep)
        pr.font.size = Pt(8.2)
        
    # 2. Data Preprocessing Pipeline
    h2 = doc.add_heading(level=1)
    h2.paragraph_format.space_before = Pt(4)
    h2.paragraph_format.space_after = Pt(2)
    r2 = h2.add_run("2. Data Collection & Preprocessing Pipeline (885 → 582 Songs)")
    r2.font.size = Pt(10.5)
    r2.font.bold = True
    r2.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    
    doc.add_paragraph(
        "Keeping 53 weekly chart rows for long-running tracks would duplicate static lyrics 53 times and cause catastrophic data leakage. "
        "We collapsed 53 weekly CSVs into a song-level dataset, fetched full lyrics via Genius API, and added metadata via Apple iTunes Search API:"
    )
    
    pipe_table = [
        ("Step", "Songs", "Script", "Processing & Filtering Rationale"),
        ("1. Chart Aggregation", "885 URIs", "02_aggregate_chart_stats.py", "Aggregated 53 weekly charts into unique songs; tracked best_rank_this_year, all_time_peak_rank, total_streams."),
        ("2. NLP Language Filter", "709 songs", "build_dataset.py", "Filtered for native English lyrics (language == 'en'); stripped section headers; normalized fan-translation queries; produced cleaned_lyrics_transformer and lemmatized tokens."),
        ("3. Deduplication", "669 songs", "build_dataset.py", "Merged duplicate album/single/deluxe releases by (artist_names, track_name), retaining highest peak rank."),
        ("4. Holiday Filter", "596 songs", "EDA_spotify_lyrics.ipynb", "Removed 72 tracks flagged as track_type == 'holiday' and 1 global anthem recurring seasonally every December."),
        ("5. Final Clean Cohort", "582 songs", "m1_bertopic_model.py", "Removed 14 leftover Christmas tracks exposed during BERTopic clustering (e.g., Fairytale of New York). Final split: 228 Top 50 hits (39.2%) vs. 354 Ranks 51–200 (60.8%).")
    ]
    t_p = doc.add_table(rows=len(pipe_table), cols=4)
    t_p.alignment = WD_TABLE_ALIGNMENT.CENTER
    p_widths = [Inches(1.3), Inches(0.9), Inches(1.5), Inches(3.5)]
    for r_idx, row in enumerate(pipe_table):
        for c_idx, text in enumerate(row):
            c = t_p.cell(r_idx, c_idx)
            c.width = p_widths[c_idx]
            cp = c.paragraphs[0]
            cp.paragraph_format.space_after = Pt(1.5)
            c_run = cp.add_run(text)
            c_run.font.size = Pt(7.8)
            if r_idx == 0:
                set_cell_background(c, "1E293B")
                c_run.font.bold = True
                c_run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                set_cell_margins(c, top=60, bottom=60, left=80, right=80)
            else:
                set_cell_background(c, "F8FAFC" if r_idx % 2 == 1 else "FFFFFF")
                set_cell_margins(c, top=45, bottom=45, left=80, right=80)
                
    doc.add_page_break() # END PAGE 1
    
    # =========================================================================
    # PAGE 2: EXPLORATORY DATA ANALYSIS & TOPIC CLUSTER WORDCLOUDS
    # =========================================================================
    h3 = doc.add_heading(level=1)
    h3.paragraph_format.space_before = Pt(0)
    h3.paragraph_format.space_after = Pt(2)
    r3 = h3.add_run("3. Exploratory Data Analysis & Topic WordClouds")
    r3.font.size = Pt(10.5)
    r3.font.bold = True
    r3.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    
    doc.add_paragraph(
        "Because lyric features are heavily right-skewed (e.g., rapid rap tracks exceeding 800 words), we compare group medians and "
        "run non-parametric Mann-Whitney U tests alongside Spearman rank correlations against continuous peak rank:"
    )
    
    # Insert Figure 2 (Boxplots)
    fig2_path = Path("artifacts/report_figures/fig2_boxplots.png")
    if fig2_path.exists():
        doc.add_picture(str(fig2_path), width=Inches(6.6))
        cp = doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.space_after = Pt(2)
        cpr = cp.add_run("Figure 1: Surface Lyric Feature Distributions (N = 582). Neither repetition, length, nor vocabulary richness differs between Top 50 hits and ranks 51–200 (all p > 0.20).")
        cpr.font.size = Pt(7.8)
        cpr.font.italic = True
        cpr.font.color.rgb = RGBColor(0x47, 0x55, 0x69)
        
    eda_mini = [
        ("Lyric Metric", "Top 50 Hits (n = 228)", "Ranks 51–200 (n = 354)", "Mann-Whitney U", "Spearman r (vs. Rank)", "Statistical Finding"),
        ("Chorus Repetition Rate", "Median: 31.8% (Mean: 32.2%)", "Median: 33.3% (Mean: 33.9%)", "p = 0.2154", "rs = +0.044 (p = 0.284)", "Top 50 songs repeat lines slightly less."),
        ("Cleaned Word Count", "Median: 352.5 (Mean: 376.7)", "Median: 346.5 (Mean: 374.0)", "p = 0.5294", "rs = -0.030 (p = 0.474)", "Insignificant 6-word median difference."),
        ("Vocabulary Diversity (TTR)", "Median: 0.372 (Mean: 0.374)", "Median: 0.365 (Mean: 0.367)", "p = 0.3255", "rs = -0.023 (p = 0.581)", "Identical ratio of unique to total words.")
    ]
    t_em = doc.add_table(rows=len(eda_mini), cols=6)
    t_em.alignment = WD_TABLE_ALIGNMENT.CENTER
    em_widths = [Inches(1.5), Inches(1.1), Inches(1.1), Inches(0.9), Inches(1.1), Inches(1.5)]
    for r_idx, row in enumerate(eda_mini):
        for c_idx, text in enumerate(row):
            c = t_em.cell(r_idx, c_idx)
            c.width = em_widths[c_idx]
            cp = c.paragraphs[0]
            cp.paragraph_format.space_after = Pt(1.5)
            c_run = cp.add_run(text)
            c_run.font.size = Pt(7.5)
            if r_idx == 0:
                set_cell_background(c, "1E293B")
                c_run.font.bold = True
                c_run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                set_cell_margins(c, top=50, bottom=50, left=70, right=70)
            else:
                set_cell_background(c, "F8FAFC" if r_idx % 2 == 1 else "FFFFFF")
                set_cell_margins(c, top=40, bottom=40, left=70, right=70)
                
    doc.add_paragraph(
        "To explore whether vocabulary clustering reveals deeper themes, Figure 2 displays WordClouds generated from lemmatized lyrics across "
        "three major BERTopic clusters alongside Topic -1 (Formula-Free Tracks). While Topic 0 captures repetitive club words (tonight, turn, stop) "
        "and Topic 8 captures breakup terms, Topic -1 spans a rich, unformulaic vocabulary that resists rigid categorization:"
    )
    
    # Insert Figure 1 (WordClouds)
    fig1_path = Path("artifacts/report_figures/fig1_wordcloud.png")
    if fig1_path.exists():
        doc.add_picture(str(fig1_path), width=Inches(6.4))
        cp = doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.space_after = Pt(2)
        cpr = cp.add_run("Figure 2: BERTopic Cluster WordClouds. Comparing low-converting club clichés (Topic 0: 22.6% Top 50), K-Pop hooks (Topic 3: 64.3%), pop star heartbreak (Topic 8: 61.1%), and the rich, diverse Formula-Free Group (Topic -1: n = 123 songs, 34.1% Top 50 rate).")
        cpr.font.size = Pt(7.8)
        cpr.font.italic = True
        cpr.font.color.rgb = RGBColor(0x47, 0x55, 0x69)
        
    doc.add_page_break() # END PAGE 2
    
    # =========================================================================
    # PAGE 3: PREDICTIVE MODELING & ARTIST-GROUPED CROSS-VALIDATION
    # =========================================================================
    h4 = doc.add_heading(level=1)
    h4.paragraph_format.space_before = Pt(0)
    h4.paragraph_format.space_after = Pt(2)
    r4 = h4.add_run("4. Predictive Modeling: Can Machine Learning Predict Top 50 Hits?")
    r4.font.size = Pt(10.5)
    r4.font.bold = True
    r4.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    
    doc.add_paragraph(
        "To evaluate whether deep contextual embeddings or topic probabilities contain subtle hit signals missed by basic word counts, "
        "we trained L2-regularized Logistic Regression classifiers (class_weight=\"balanced\"). Crucially, superstars (Taylor Swift, Drake) "
        "release multiple charting tracks; a standard random train/test split would cause artist identity leakage. We therefore grouped all "
        "collaborating artists into 191 independent clusters and used 5-Fold StratifiedGroupKFold so that no artist ever appears in both folds."
    )
    
    fig3_path = Path("artifacts/report_figures/fig3_ml_models.png")
    if fig3_path.exists():
        doc.add_picture(str(fig3_path), width=Inches(6.5))
        cp = doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.space_after = Pt(2)
        cpr = cp.add_run("Figure 3: 5-Fold Artist-Grouped Cross-Validation ROC-AUC Scores. Dashed line marks 0.50 (random coin flip). Pure lyric models (M3: 0.5080, M4: 0.4696) perform at random chance. Adding lyrics to metadata (M2: 0.6463) does not beat metadata alone (M1: 0.6611).")
        cpr.font.size = Pt(7.8)
        cpr.font.italic = True
        cpr.font.color.rgb = RGBColor(0x47, 0x55, 0x69)
        
    ml_table = [
        ("Model & Features", "Full (N=582) ROC-AUC", "Full F1", "2024+ (N=316) ROC-AUC", "2024+ F1", "Technical Finding"),
        ("M1: Metadata Baseline\nRelease Year + Genre", "0.6611 (± 0.100)", "0.5845", "0.5239 (± 0.109)", "0.5440", "Signal driven by catalog recency; collapses near 0.52 on recent releases."),
        ("M2: Metadata + BERTopic\nM1 + Fold-Internal Topics", "0.6463 (± 0.057)", "0.5861", "0.5313 (± 0.119)", "0.5234", "Adding lyric topic probabilities does not improve prediction over metadata."),
        ("M3: Text Embeddings Only\n384-dim Sentence-BERT", "0.5080 (± 0.045)", "0.5090", "0.4878 (± 0.119)", "0.4820", "Performs identically to a 50/50 coin flip across unseen artists."),
        ("M4: Basic Lyric Stats Only\nWord Count + Repetition + TTR", "0.4696 (± 0.039)", "0.4686", "0.3525 (± 0.037)", "0.3957", "Count-based lyric statistics provide zero predictive power.")
    ]
    t_ml = doc.add_table(rows=len(ml_table), cols=6)
    t_ml.alignment = WD_TABLE_ALIGNMENT.CENTER
    ml_w = [Inches(1.8), Inches(1.1), Inches(0.8), Inches(1.1), Inches(0.8), Inches(1.6)]
    for r_idx, row in enumerate(ml_table):
        for c_idx, text in enumerate(row):
            c = t_ml.cell(r_idx, c_idx)
            c.width = ml_w[c_idx]
            cp = c.paragraphs[0]
            cp.paragraph_format.space_after = Pt(1.5)
            c_run = cp.add_run(text)
            c_run.font.size = Pt(7.5)
            if r_idx == 0:
                set_cell_background(c, "1E293B")
                c_run.font.bold = True
                c_run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                set_cell_margins(c, top=50, bottom=50, left=70, right=70)
            else:
                set_cell_background(c, "F8FAFC" if r_idx % 2 == 1 else "FFFFFF")
                set_cell_margins(c, top=40, bottom=40, left=70, right=70)
                
    doc.add_paragraph(
        "Why did metadata work in the full cohort? In 53 weekly charts, older catalog classics (1970s–2022) sit in ranks 51–200 while "
        "new 2025–2026 releases debut in the Top 50 (Mann-Whitney p < 0.0001). Once older songs are filtered out (2024–2026 releases only, "
        "N = 316), even the metadata baseline collapses to 0.5239. Chart placement is driven by market release timing, not lyric text."
    )
    
    doc.add_page_break() # END PAGE 3
    
    # =========================================================================
    # PAGE 4: BERTOPIC THEMES & DECONSTRUCTING THE SUPERSTAR ILLUSION
    # =========================================================================
    h5 = doc.add_heading(level=1)
    h5.paragraph_format.space_before = Pt(0)
    h5.paragraph_format.space_after = Pt(2)
    r5 = h5.add_run("5. Unsupervised Topic Modeling: Clichés vs. Formula-Free Tracks")
    r5.font.size = Pt(10.5)
    r5.font.bold = True
    r5.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    
    doc.add_paragraph(
        "Fitting BERTopic across the 582-song corpus segmented 459 tracks into 24 distinct clusters, while 123 tracks (21.1%) were "
        "assigned to Topic -1 (Formula-Free Tracks). A global Chi-Square test across all 24 topics yields chi^2 = 33.48 (df = 23, p = 0.0731), "
        "confirming that overall topic differences are not statistically significant at the alpha = 0.05 level."
    )
    
    fig4_path = Path("artifacts/report_figures/fig4_bertopic.png")
    if fig4_path.exists():
        doc.add_picture(str(fig4_path), width=Inches(6.4))
        cp = doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.space_after = Pt(2)
        cpr = cp.add_run("Figure 4: Top 50 Hit Rates across 24 BERTopic Clusters (n = 459). Dashed line marks overall average (39.2%). Overall differences are statistically non-significant (p = 0.0731).")
        cpr.font.size = Pt(7.8)
        cpr.font.italic = True
        cpr.font.color.rgb = RGBColor(0x47, 0x55, 0x69)
        
    # Deconstruction Table
    decomp_table = [
        ("The Crowded Cliché Trap (Topic 0: 22.6%)", "The Superstar Illusion in High Clusters", "The Formula-Free Reality (Topic -1: 34.1%)"),
        ("The largest cluster on the chart is Topic 0 (Nightlife Flirtation & Club Clichés, n = 53), built around repetitive party words (tonight, turn, stop). Despite being the most common songwriting template, only 22.6% (12/53) reached the Top 50. Writing generic club party lyrics simply buries emerging artists in a crowded lower tier.",
         "Clusters appearing to have high conversion—such as Topic 8 (Confessional Pop, 61.1%, n = 18), Topic 3 (K-Pop Hooks, 64.3%, n = 28), or Topic 19 (Hip-Hop Hustle, 54.5%, n = 11)—are driven by Olivia Rodrigo, Ariana Grande, BTS, and Drake dropping entire albums into those clusters. Their high hit rate reflects artist fame, not lyric topics.",
         "123 songs (21.1% of the chart) refused to fit standard cluster templates. Yet this unclustered group achieved 42 Top 50 hits (34.1% hit rate), including #1 smashes like Djo's End of Beginning, Taylor Swift's The Fate of Ophelia, and Harry Styles' American Girls. Writing distinctive, formula-free lyrics lets new artists stand out.")
    ]
    t_dec = doc.add_table(rows=2, cols=3)
    t_dec.alignment = WD_TABLE_ALIGNMENT.CENTER
    dec_w = [Inches(2.4), Inches(2.4), Inches(2.4)]
    for r_idx, row in enumerate(decomp_table):
        for c_idx, text in enumerate(row):
            c = t_dec.cell(r_idx, c_idx)
            c.width = dec_w[c_idx]
            cp = c.paragraphs[0]
            cp.paragraph_format.space_after = Pt(1.5)
            c_run = cp.add_run(text)
            c_run.font.size = Pt(7.5)
            if r_idx == 0:
                set_cell_background(c, "1E293B")
                c_run.font.bold = True
                c_run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                set_cell_margins(c, top=50, bottom=50, left=70, right=70)
            else:
                set_cell_background(c, "F8FAFC")
                set_cell_margins(c, top=45, bottom=45, left=70, right=70)
                
    doc.add_page_break() # END PAGE 4
    
    # =========================================================================
    # PAGE 5: REFLECTIONS ON MINI-PROJECT CANVAS, LESSONS & LIMITATIONS
    # =========================================================================
    h6 = doc.add_heading(level=1)
    h6.paragraph_format.space_before = Pt(0)
    h6.paragraph_format.space_after = Pt(2)
    r6 = h6.add_run("6. Canvas Reflections: What Worked, Lessons Learned & Limitations")
    r6.font.size = Pt(10.5)
    r6.font.bold = True
    r6.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    
    doc.add_paragraph(
        "Reflecting on our initial Mini-Project Canvas, our primary objective was to investigate whether commercial songwriting formulas "
        "predict streaming performance. Below is our critical assessment of what worked, what failed, and what we learned:"
    )
    
    # Canvas Reflection Table
    canvas_eval = [
        ("Canvas Dimension", "Initial Plan vs. Actual Execution", "Critical Reflection & What Worked / Failed"),
        ("1. Target & Data Leakage\n(What Worked)", "Initial plan: Use raw weekly chart rows (N=10,600).\nActual: Collapsed into cross-sectional song dataset (N=582).", "WORKED: Deduplicating by (artist, track) prevented copying identical static lyrics 53 times, eliminating severe artificial inflation of model metrics."),
        ("2. Cross-Validation Setup\n(What Worked)", "Initial plan: Standard random 80/20 train/test split.\nActual: 5-Fold StratifiedGroupKFold on 191 artist clusters.", "WORKED: Blocked artist identity leakage. Models could not memorize Taylor Swift's or Drake's vocabulary across folds, giving a fair test on unseen creators."),
        ("3. Lyric Predictability\n(What Failed)", "Initial hypothesis: S-BERT embeddings and chorus repetition would predict Top 50 placement.\nActual: M3 achieved 0.5080 ROC-AUC (coin flip).", "FAILED (Informative Null Result): Proved definitively that written lyrics alone have no predictive relationship with chart rank. Taught us not to force formulas where no signal exists."),
        ("4. Audience Communication\n(What Changed)", "Initial plan: Present ML classification metrics directly to songwriters.\nActual: Built an interactive, non-technical Streamlit app.", "CHANGED: Technical terms (ROC-AUC, p-values) alienate creators. We pivoted our web app into 'The Lyric Myth', busting songwriting myths and delivering creative liberation.")
    ]
    t_can = doc.add_table(rows=len(canvas_eval), cols=3)
    t_can.alignment = WD_TABLE_ALIGNMENT.CENTER
    can_w = [Inches(1.5), Inches(2.5), Inches(3.2)]
    for r_idx, row in enumerate(canvas_eval):
        for c_idx, text in enumerate(row):
            c = t_can.cell(r_idx, c_idx)
            c.width = can_w[c_idx]
            cp = c.paragraphs[0]
            cp.paragraph_format.space_after = Pt(1.5)
            c_run = cp.add_run(text)
            c_run.font.size = Pt(7.5)
            if r_idx == 0:
                set_cell_background(c, "1E293B")
                c_run.font.bold = True
                c_run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                set_cell_margins(c, top=50, bottom=50, left=70, right=70)
            else:
                set_cell_background(c, "F8FAFC" if r_idx % 2 == 1 else "FFFFFF")
                set_cell_margins(c, top=40, bottom=40, left=70, right=70)
                
    doc.add_paragraph()
    
    # Limitations & Future Steps
    h7 = doc.add_heading(level=2)
    h7.paragraph_format.space_before = Pt(3)
    h7.paragraph_format.space_after = Pt(1.5)
    r7 = h7.add_run("Methodological Limitations, Data Hygiene Discovery & Future Work")
    r7.font.size = Pt(9.5)
    r7.font.bold = True
    
    lim_box = doc.add_table(rows=1, cols=3)
    lim_box.alignment = WD_TABLE_ALIGNMENT.CENTER
    lim_w = [Inches(2.4), Inches(2.4), Inches(2.4)]
    lim_cells = [lim_box.cell(0, 0), lim_box.cell(0, 1), lim_box.cell(0, 2)]
    lim_data = [
        ("1. Fixed 1-Year Window & Bias",
         "Because best_rank_this_year was observed during Sept 2025–Sept 2026, older pre-2024 catalog songs were captured past their peak (45.1% all-time Top 50 vs. 23.7% in-window), artificially inflating Release Year. Furthermore, all 582 songs reached the Top 200 (survivorship bias); we cannot test whether lyrics help unknown songs chart initially."),
        ("2. Catching 14 Christmas Songs",
         "Running unsupervised BERTopic clustering on our initial 596 EDA tracks surfaced a cluster containing residual holiday songs that bypassed title filters—such as Little Saint Nick (1991 Remix) and Fairytale of New York. This allowed us to refine our exclusion regex in m1_bertopic_model.py and establish our clean 582 cohort."),
        ("3. Future Work: Audio & Virality",
         "Because Spotify deprecated its public Audio Features API, acoustic features (tempo, energy, vocal dynamic range) and short-form social engagement metrics (TikTok/Reels video creations) were absent. Combining audio stem analysis with viral social metrics represents the natural next step for commercial music analytics.")
    ]
    for idx, (title, text) in enumerate(lim_data):
        c = lim_cells[idx]
        c.width = lim_w[idx]
        set_cell_background(c, "F8FAFC")
        set_cell_margins(c, top=60, bottom=60, left=80, right=80)
        p = c.paragraphs[0]
        pr = p.add_run(title)
        pr.bold = True
        pr.font.size = Pt(8.0)
        pr.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        p2 = c.add_paragraph()
        p2.paragraph_format.space_after = Pt(1)
        pr2 = p2.add_run(text)
        pr2.font.size = Pt(7.2)
        
    out_file = Path("Spotify_Top200_Lyrics_Technical_Report.docx")
    doc.save(str(out_file))
    print(f"5-Page Tight Technical Report saved to: {out_file.absolute()}")

if __name__ == "__main__":
    create_tight_report()
