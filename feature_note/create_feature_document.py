"""Create the final model's feature list from an explicit, verified contract."""
from pathlib import Path
import json
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

root=Path(__file__).resolve().parents[1]
groups=[('Demographics and genetics',[
('AGE','Age at the baseline assessment.'),
('PTGENDER','Recorded participant sex or gender category.'),
('PTEDUCAT','Years of formal education.'),
('APOE4','Number of APOE epsilon 4 alleles, recorded as 0, 1, or 2.')]),
('Cognitive assessments',[
('CDRSB_bl','Clinical Dementia Rating sum of boxes.'),
('ADAS11_bl','11 item Alzheimer\u2019s Disease Assessment Scale cognitive score.'),
('ADAS13_bl','13 item Alzheimer\u2019s Disease Assessment Scale cognitive score.'),
('MMSE_bl','Mini Mental State Examination total score.'),
('RAVLT_immediate_bl','Rey Auditory Verbal Learning Test immediate recall score.'),
('RAVLT_learning_bl','RAVLT learning score across repeated trials.'),
('RAVLT_forgetting_bl','RAVLT forgetting score.'),
('RAVLT_perc_forgetting_bl','RAVLT percentage forgetting score.'),
('FAQ_bl','Functional Activities Questionnaire score.'),
('MOCA_bl','Montreal Cognitive Assessment total score.')]),
('Cerebrospinal fluid biomarkers',[
('ABETA_bl','CSF amyloid beta measurement.'),
('TAU_bl','CSF total tau measurement.'),
('PTAU_bl','CSF phosphorylated tau measurement.')]),
('MRI derived measurements',[
('Hippocampus_bl','Hippocampal volume.'),
('Ventricles_bl','Ventricular volume.'),
('WholeBrain_bl','Whole brain volume.'),
('Entorhinal_bl','Entorhinal cortex volume.'),
('Fusiform_bl','Fusiform region volume.'),
('MidTemp_bl','Middle temporal region volume.'),
('ICV_bl','Intracranial volume.')]),
('PET measurement',[
('FDG_bl','FDG PET composite measure of brain glucose metabolism.')]),
('Medical history',[
('MH4CARD','Cardiovascular history flag.'),
('MH9ENDO','Endocrine or metabolic history flag.'),
('MH16SMOK','Smoking history flag.')])]
fields=[field for _,rows in groups for field,_ in rows]
assert len(fields)==28 and len(set(fields))==28
manifest=json.loads((root/'local/frozen_manifest.json').read_text())
assert set(fields)=={c for c in manifest['fitted_features'] if not c.endswith('_bound')}

doc=Document();sec=doc.sections[0]
sec.page_width=Inches(8.5);sec.page_height=Inches(11)
sec.top_margin=sec.bottom_margin=Inches(.65)
sec.left_margin=sec.right_margin=Inches(.7)
for name in ['Normal','Title','Heading 1','Heading 2']:
 style=doc.styles[name];style.font.name='Calibri';style.font.color.rgb=RGBColor(0,0,0)
for border in list(doc.styles.element.iter(qn('w:pBdr'))):
 border.getparent().remove(border)
normal=doc.styles['Normal'];normal.font.size=Pt(10.5)
normal.paragraph_format.space_after=Pt(5)
normal.paragraph_format.line_spacing=1.03
doc.styles['Title'].font.size=Pt(21);doc.styles['Title'].paragraph_format.space_after=Pt(8)
doc.styles['Heading 1'].font.size=Pt(12)
doc.styles['Heading 1'].paragraph_format.space_before=Pt(10)
doc.styles['Heading 1'].paragraph_format.space_after=Pt(4)
doc.add_paragraph('Features used in the final BTP prediction model','Title')
p=doc.add_paragraph();p.add_run('We extracted 60 baseline features for analysis. ').bold=True
p.add_run('After comparing feature panels and models, the selected logistic regression model uses the following 28 core source features to predict dementia conversion within five years.')
doc.add_paragraph('The suffix _bl identifies a baseline measurement. MRI and PET entries are numeric features derived from imaging; raw scan files are not model inputs.')

def table(rows):
 t=doc.add_table(rows=1,cols=2);t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
 t.columns[0].width=Inches(2.55);t.columns[1].width=Inches(4.55)
 for c,txt in zip(t.rows[0].cells,['Feature column','Meaning']):
  c.text=txt;shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E7EBEF');c._tc.get_or_add_tcPr().append(shade)
  for r in c.paragraphs[0].runs:r.bold=True
 repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
 for field,description in rows:
  row=t.add_row();row.cells[0].text=field;row.cells[1].text=description
  for r in row.cells[0].paragraphs[0].runs:r.font.name='Consolas';r.font.size=Pt(9.5)
 for row in t.rows:
  prevent=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(prevent)
  for index,cell in enumerate(row.cells):
   cell.width=Inches(2.55 if index==0 else 4.55)
   cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   pr=cell._tc.get_or_add_tcPr();margins=OxmlElement('w:tcMar')
   for edge,val in [('top',55),('bottom',55),('left',85),('right',85)]:
    x=OxmlElement('w:'+edge);x.set(qn('w:w'),str(val));x.set(qn('w:type'),'dxa');margins.append(x)
   pr.append(margins)
   for p in cell.paragraphs:p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.0
 borders=OxmlElement('w:tblBorders')
 for edge in ['top','left','bottom','right','insideH','insideV']:
  x=OxmlElement('w:'+edge);x.set(qn('w:val'),'single');x.set(qn('w:sz'),'4');x.set(qn('w:color'),'D9D9D9');borders.append(x)
 t._tbl.tblPr.append(borders)

for i,(name,rows) in enumerate(groups):
 if i==2:doc.add_page_break()
 doc.add_paragraph(name+' '+str(len(rows))+(' feature' if len(rows)==1 else ' features'),'Heading 1');table(rows)

doc.add_paragraph('Feature count and interpretation','Heading 1')
doc.add_paragraph('The 28 features comprise 4 demographic or genetic fields, 10 cognitive scores, 3 CSF biomarkers, 7 MRI measurements, 1 PET measurement, and 3 history indicators.')
doc.add_paragraph('Medical history is aggregated from records at or before baseline. A documented condition is coded 1, documented absence is coded 0, and unavailable history remains missing for preprocessing.')
doc.add_paragraph('These counts refer to original input features, not learned regression coefficients. The saved pipeline also retains two laboratory bound indicators; missingness indicators and category encoding expand the final model input to 60 transformed columns. This is separate from the 60 source features originally extracted.')
doc.add_paragraph('Verification source: the implemented feature registry and frozen model manifest in the project repository. Participant identifiers, diagnosis labels, and follow-up dates are excluded from model predictors.')
footer=sec.footer.paragraphs[0];footer.alignment=2
r=footer.add_run('BTP selected features  |  ');r.font.size=Pt(9)
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
out=root.parent/'output/documents/BTP_Selected_Features.docx';out.parent.mkdir(parents=True,exist_ok=True)
doc.save(out);print(out)
