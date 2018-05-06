# -*- coding: utf-8 -*-
# Generated by Django 1.10.1 on 2018-05-06 09:49
from __future__ import unicode_literals

import datetime
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('records', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Presensi',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('tanggal', models.DateField(blank=True, null=True)),
                ('jam_masuk', models.TimeField(blank=True, null=True)),
                ('jam_pulang', models.TimeField(blank=True, null=True)),
                ('kehadiran', models.IntegerField(blank=True, choices=[(0, 'Alpha'), (1, 'Hadir'), (2, 'Izin'), (3, 'Sakit')], null=True)),
                ('recorded_at', models.DateTimeField(blank=True, default=datetime.datetime.now)),
            ],
        ),
        migrations.AlterModelOptions(
            name='records',
            options={'verbose_name_plural': 'Records'},
        ),
        migrations.RemoveField(
            model_name='records',
            name='bio',
        ),
        migrations.RemoveField(
            model_name='records',
            name='country',
        ),
        migrations.RemoveField(
            model_name='records',
            name='education',
        ),
        migrations.RemoveField(
            model_name='records',
            name='first_name',
        ),
        migrations.RemoveField(
            model_name='records',
            name='last_name',
        ),
        migrations.RemoveField(
            model_name='records',
            name='marital_status',
        ),
        migrations.RemoveField(
            model_name='records',
            name='occupation',
        ),
        migrations.RemoveField(
            model_name='records',
            name='residence',
        ),
        migrations.AddField(
            model_name='records',
            name='agama',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AddField(
            model_name='records',
            name='alamat',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='records',
            name='jabatan',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='records',
            name='jenis_kelamin',
            field=models.IntegerField(blank=True, choices=[(0, 'Laki-laki'), (1, 'Perempuan')], null=True),
        ),
        migrations.AddField(
            model_name='records',
            name='nama',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AddField(
            model_name='records',
            name='npp',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AddField(
            model_name='records',
            name='status_pegawai',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AddField(
            model_name='records',
            name='tanggal_lahir',
            field=models.DateField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='records',
            name='tempat_lahir',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='records',
            name='id',
            field=models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID'),
        ),
        migrations.AddField(
            model_name='presensi',
            name='pegawai',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='presensi', to='records.Records'),
        ),
    ]