"""
PDF Report Generator Module for AI Investor SaaS
Creates downloadable PDF reports with analysis and charts
"""

from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
import io
import plotly.graph_objects as go

def create_pdf_report(stocks_data, ai_analysis, output_filename):
    """
    Create a professional PDF report
    
    Args:
        stocks_data: Dictionary of stock data
        ai_analysis: AI-generated analysis text
        output_filename: Output file path/buffer
    
    Returns:
        BytesIO buffer containing the PDF
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#9370DB'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#9370DB'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    )
    
    # Title Page
    elements.append(Spacer(1, 2*inch))
    title = Paragraph("📈 AI Holistic Investor Report", title_style)
    elements.append(title)
    elements.append(Spacer(1, 12))
    
    subtitle = Paragraph("Comprehensive Fundamental + Technical Analysis", styles['Normal'])
    subtitle.alignment = TA_CENTER
    elements.append(subtitle)
    elements.append(Spacer(1, 0.5*inch))
    
    # Report metadata
    metadata_style = ParagraphStyle('metadata', parent=styles['Normal'], alignment=TA_CENTER, fontSize=9)
    
    stock_symbols = ", ".join(stocks_data.keys())
    report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    
    elements.append(Paragraph(f"<b>Analyzed Stocks:</b> {stock_symbols}", metadata_style))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(f"<b>Generated:</b> {report_date}", metadata_style))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph("<b>Powered by:</b> Perplexity AI", metadata_style))
    
    elements.append(PageBreak())
    
    # AI Analysis Section
    elements.append(Paragraph("AI Investment Analysis", heading_style))
    elements.append(Spacer(1, 12))
    
    # Split analysis into paragraphs
    analysis_paragraphs = ai_analysis.split('\n\n')
    for para in analysis_paragraphs:
        if para.strip():
            # Clean up markdown-style formatting for PDF
            cleaned_para = para.replace('**', '<b>').replace('**', '</b>')
            cleaned_para = cleaned_para.replace('##', '').replace('#', '')
            elements.append(Paragraph(cleaned_para, body_style))
            elements.append(Spacer(1, 6))
    
    elements.append(PageBreak())
    
    # Stock Summary Table
    elements.append(Paragraph("Portfolio Summary", heading_style))
    elements.append(Spacer(1, 12))
    
    # Create summary table
    table_data = [['Symbol', 'Current Price', 'SMA 50', 'RSI', 'Recommendation']]
    
    for symbol, data in stocks_data.items():
        technical = data.get('technical', {}).get('summary', {})
        info = data.get('info', {})
        
        current_price = f"₹{technical.get('current_price', 'N/A'):.2f}" if technical.get('current_price') else 'N/A'
        sma_50 = f"₹{technical.get('sma_50', 'N/A'):.2f}" if technical.get('sma_50') else 'N/A'
        rsi = f"{technical.get('rsi_14', 'N/A'):.1f}" if technical.get('rsi_14') else 'N/A'
        
        # Simple recommendation logic
        rec = "N/A"
        if technical.get('current_price') and technical.get('sma_50'):
            if technical['current_price'] > technical['sma_50']:
                rec = "Bullish"
            else:
                rec = "Bearish"
        
        table_data.append([symbol, current_price, sma_50, rsi, rec])
    
    table = Table(table_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9370DB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 20))
    
    # Disclaimer
    elements.append(PageBreak())
    disclaimer_style = ParagraphStyle('disclaimer', parent=styles['Normal'], fontSize=8, alignment=TA_JUSTIFY)
    elements.append(Paragraph("<b>DISCLAIMER</b>", heading_style))
    elements.append(Paragraph(
        "This report is generated by AI for informational purposes only and should not be considered as financial advice. "
        "Past performance is not indicative of future results. Please consult with a qualified financial advisor before making "
        "any investment decisions. The data presented is based on publicly available information and may contain errors or omissions. "
        "The creators of this report are not responsible for any investment decisions made based on this analysis.",
        disclaimer_style
    ))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

def export_chart_as_image(fig, width=800, height=600):
    """
    Export plotly figure as image bytes
    
    Args:
        fig: Plotly figure object
        width: Image width
        height: Image height
    
    Returns:
        BytesIO buffer containing the image
    """
    img_bytes = fig.to_image(format="png", width=width, height=height, engine="kaleido")
    return io.BytesIO(img_bytes)
