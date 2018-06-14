from django import forms

from records.models import Records, Presensi, Citra

class RecordsForm(forms.ModelForm):
    class Meta:
        model = Records
        fields = ('npp','nama','tempat_lahir','tanggal_lahir','jenis_kelamin','alamat','agama','status_pegawai','jabatan')

class PresensiForm(forms.ModelForm):
    class Meta:
        model = Presensi
        fields = ('pegawai','tanggal','jam_masuk','jam_pulang','kehadiran')

class CitraForm(forms.ModelForm):
    class Meta:
        model = Citra
        fields = ('npp','citra_name','pegawai')