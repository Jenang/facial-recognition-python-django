# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from datetime import datetime
from django.db import models

# Create your models here.
class Records(models.Model):

    GENDER = (
        (0, 'Laki-laki'),
        (1, 'Perempuan')
    )

    npp = models.CharField(max_length=100, null=True, blank=True, unique=True)
    nama = models.CharField(max_length=100, null=True, blank=True)
    foto_profile = models.ImageField(null=True, blank=True)
    tempat_lahir = models.CharField(max_length=100, null=True, blank=True)
    tanggal_lahir = models.DateField(null=True, blank=True)
    jenis_kelamin = models.IntegerField(choices=GENDER, blank=True, null=True)
    alamat = models.TextField(null=True, blank=True)
    agama = models.CharField(max_length=100, null=True, blank=True)
    status_pegawai = models.CharField(max_length=100, null=True, blank=True)
    jabatan = models.TextField(null=True, blank=True)
    recorded_at = models.DateTimeField(default=datetime.now, blank=True)

    def __str__(self):
        return self.nama
    class Meta:
        verbose_name_plural = "Records"

class Presensi(models.Model):

    ABSENSI = (
        (0, 'Alpha'),
        (1, 'Hadir'),
        (2, 'Izin'),
        (3, 'Sakit')
    )

    pegawai = models.ForeignKey(Records, null=True, blank=True, related_name='presensi')
    tanggal = models.DateField(null=True, blank=True)
    jam_masuk = models.TimeField(null=True, blank=True)
    jam_pulang = models.TimeField(null=True, blank=True)
    kehadiran = models.IntegerField(choices=ABSENSI, blank=True, null=True)
    recorded_at = models.DateTimeField(default=datetime.now, blank=True)

class Citra(models.Model):

    pegawai = models.ForeignKey(Records, null=True, blank=True, related_name='citra')
    npp = models.CharField(max_length=100, null=True, blank=True)
    citra_name = models.CharField(max_length=100, null=True, blank=True)