from django import forms
from gmapi import forms as gmap_forms

class TSPForm(forms.Form):
    Length = forms.IntegerField(min_value=1, max_value=100, label="Grid size X")
    Width = forms.IntegerField(min_value=1, max_value=100, label="Grid size Y")
    algorithm = forms.ChoiceField(choices=[('nn', 'Nearest Neighbor'), ('two-opt', '2-opt'), ('christofides', 'Christofides')])