from pathlib import Path
import json
from docx import Document
from docx.shared import Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE

root=Path(__file__).resolve().parent
summary=json.loads((root/'datasets/plantnet-300k-v2/balanced-75000/summary.json').read_text())
doc=Document(root/'Dataset_Sources.docx')
for p in doc.paragraphs:
    if p.text.startswith('Τα δεδομένα αποθηκεύτηκαν στον φάκελο'):
        p.text='Τα δεδομένα αποθηκεύτηκαν στον φάκελο deep-learning-course/plant-health-project. Το PlantVillage υποστηρίζει αναγνώριση καλλιέργειας και κατάστασης φύλλου, το Oxford Flowers 102 αναγνώριση άνθους και το Pl@ntNet αναγνώριση είδους φυτού. Οι πηγές δεν παρέχουν τεκμηριωμένη θεραπεία ανά φωτογραφία.'
    elif p.text.startswith('Οι ακόλουθες πηγές καταγράφηκαν'):
        p.text='Πρόσθετες πηγές και κατάσταση λήψης. Η επιλεγμένη υποομάδα του Pl@ntNet περιγράφεται αναλυτικά στην επόμενη ενότητα.'
    elif p.text.startswith('Η επίσημη πηγή περιγράφει 32 είδη'):
        p.text='Η επίσημη πηγή περιγράφει 32 είδη και αντιστοίχιση ονομάτων με αριθμούς εικόνων. Διατηρήθηκε το source-labels.html. Η λήψη εικόνων απέτυχε: ο διακομιστής επέστρεψε HTML. Το μη έγκυρο Leaves.tar.bz2 διαγράφηκε κατόπιν εντολής του χρήστη. Δεν υπάρχει χρησιμοποιήσιμο dataset Flavia στον φάκελο.'
    elif p.text.startswith('Pl@ntNet 300K v2:'):
        p.text='Pl@ntNet 300K v2: η πλήρης πηγή περιέχει 306.087 εικόνες και 1.000 είδη. Δημιουργήθηκε υποσύνολο 75.000 εικόνων με όλα τα είδη και την ελάχιστη διασπορά που επιτρέπουν τα ήδη αποθηκευμένα αρχεία. Βλ. την επόμενη ενότητα για πλήθη και χρήση.'

# Re-running refreshes only our appendix.
body=doc._element.body
for element in list(body):
    if ''.join(element.itertext()).startswith(('Pl@ntNet 80000','Pl@ntNet 75000')):
        previous=element.getprevious()
        if previous is not None and previous.findall('.//'+qn('w:br')):
            body.remove(previous)
        found=False
        for e in list(body):
            if e is element:found=True
            if found and e.tag!=qn('w:sectPr'):body.remove(e)
        break
doc.add_page_break()
doc.add_heading('Pl@ntNet 75000 για εκπαίδευση και έλεγχο',1)
def p(s):doc.add_paragraph(s)
complete=summary['status']=='complete'
p('Ενημέρωση 27 Σεπτεμβρίου 2026. Η λήψη σταμάτησε κατόπιν εντολής του χρήστη στις 79.831 εικόνες. Από αυτές δημιουργήθηκε έτοιμο, επαληθευμένο υποσύνολο ακριβώς 75.000 εικόνων. Οι επιπλέον 4.831 εικόνες διατηρούνται στον αρχικό φάκελο.')
p('Θέση: datasets/plantnet-300k-v2/balanced-75000 μέσα στο plant-health-project. Οι εικόνες βρίσκονται σε images/train, images/val και images/test, σε υποφακέλους 0000–0999. Χρησιμοποιούνται hard links χωρίς διπλασιασμό χώρου· επεξεργασία μιας εικόνας αλλάζει και το αντίστοιχο αρχικό αρχείο.')
doc.add_heading('Περιεχόμενο και ισορροπία',2)
p(f"75.000 εικόνες από 1.000 είδη. Εκπαίδευση: {summary['split_counts']['train']:,} · επικύρωση: {summary['split_counts']['val']:,} · τελικός έλεγχος: {summary['split_counts']['test']:,}. Είδη ανά split: {summary['species_per_split']['train']}, {summary['species_per_split']['val']}, {summary['species_per_split']['test']} αντίστοιχα. Ανά είδος: ελάχιστο {summary['min_per_species']}, μέγιστο {summary['max_per_species']}, μέσος όρος 75, πληθυσμιακή τυπική απόκλιση {summary['population_stddev']:.3f}.")
p(f"Η κατανομή ελαχιστοποιεί τη διασπορά με περιορισμό το αποθηκευμένο πλήθος κάθε είδους. Κρατήθηκαν όλες οι αποθηκευμένες εικόνες {summary['species_at_local_capacity']} ειδών. Οι υπόλοιπες ποσοστώσεις διαφέρουν το πολύ κατά μία εικόνα. Δεν προστέθηκαν διπλότυπα ή τεχνητές εικόνες.")
p('Διατηρήθηκε το αρχικό split κάθε εικόνας, με αναλογική κατανομή ποσοστώσεων και μεγαλύτερα υπόλοιπα. Η αρχική λήψη χρησιμοποίησε τυχαία κυκλικά τμήματα της σειράς ZIP. Η τελική επιλογή των 75.000 έγινε ομοιόμορφα χωρίς επανάθεση μέσα στις αποθηκευμένες ομάδες είδους/split, με seed 20260927. Δεν αποτελεί ομοιόμορφο δείγμα όλης της αρχικής πηγής. Δεν υπάρχουν κοινά observation IDs μεταξύ των επιλεγμένων splits.')
doc.add_heading('Πρακτική χρήση',2)
p('Τα train.csv, val.csv και test.csv συνδέουν διαδρομές εικόνων με species_id, επιστημονικό όνομα, όργανο, observation ID, φωτογράφο και άδεια. Το species_counts.csv περιλαμβάνει πλήθη και αντιστοίχιση κλάσεων. Το manifest.csv περιλαμβάνει και τις αρχικές διαδρομές και CRC32 του ZIP.')
p('Το plantnet_data.py παρέχει load_split και load_my_photo για TensorFlow: εικόνες RGB 224 × 224 με διατήρηση αναλογιών και τιμές [0,1]. Χρησιμοποιήστε σταθερές ετικέτες 0–999 και μοντέλο με 1.000 εξόδους. Αναλυτικό παράδειγμα υπάρχει στο PLANTNET_QUICKSTART.md. Φωτογραφίες δικών σας φυτών χρειάζονται γνωστές ετικέτες για μέτρηση ακρίβειας.')
p('Δεν υπάρχουν ετικέτες θεραπείας, ποτίσματος ή κατηγορίας δέντρο/θάμνος/ποώδες. Οι άδειες διαφέρουν ανά φωτογραφία και διατηρούνται στο CSV. Απαιτείται αναφορά στους δημιουργούς και τήρηση της αντίστοιχης άδειας.')
doc.add_heading('Προέλευση και επαλήθευση',2)
p('Πηγή: Zenodo, Pl@ntNet-300K-v2, DOI 10.5281/zenodo.10419064. Αναφορά: Garcin et al. (2021), NeurIPS Datasets and Benchmarks. Η λήψη γίνεται με HTTP byte ranges από το αρχικό ZIP, χωρίς λήψη ολόκληρου του αρχείου 41,8 GB.')
for label,url in [('Εγγραφή πηγής','https://zenodo.org/records/10419064'),('Ακριβής πηγή εικόνων','https://zenodo.org/api/records/10419064/files/images.zip/content')]:
    para=doc.add_paragraph(label+': '); h=OxmlElement('w:hyperlink');h.set(qn('r:id'),para.part.relate_to(url,RELATIONSHIP_TYPE.HYPERLINK,is_external=True));r=OxmlElement('w:r');t=OxmlElement('w:t');t.text=url;r.append(t);h.append(r);para._p.append(h)
p('Τα αρχεία plantnet300K_metadata.csv, species_metadata.csv και README.md προέρχονται από την ίδια διεύθυνση API, αντικαθιστώντας το images.zip με το αντίστοιχο όνομα. Τα MD5 τους ελέγχθηκαν έναντι της εγγραφής Zenodo. '+('Κάθε αποθηκευμένη εικόνα πέρασε έλεγχο μεγέθους και CRC32 έναντι του ZIP.' if complete else 'Ο downloader ελέγχει μέγεθος και CRC32 κάθε εικόνας πριν ολοκληρωθεί η λήψη.'))
for name in ['Title','Heading 1','Heading 2']:
    doc.styles[name].font.color.rgb=RGBColor(0,0,0)
for tree in [doc._element,doc.styles.element]:
    for border in list(tree.iter(qn('w:pBdr'))):border.getparent().remove(border)
doc.save(root/'Dataset_Sources.docx')
print('Updated Word sources; download status:',summary['status'])
