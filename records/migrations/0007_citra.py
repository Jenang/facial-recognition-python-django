# -*- coding: utf-8 -*-
# Generated by Django 1.10.1 on 2018-06-14 05:34
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('records', '0006_auto_20180507_1717'),
    ]

    operations = [
        migrations.CreateModel(
            name='Citra',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('npp', models.CharField(blank=True, max_length=100, null=True)),
                ('citra_name', models.CharField(blank=True, max_length=100, null=True)),
                ('pegawai', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='citra', to='records.Records')),
            ],
        ),
    ]