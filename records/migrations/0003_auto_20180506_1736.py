# -*- coding: utf-8 -*-
# Generated by Django 1.10.1 on 2018-05-06 10:36
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('records', '0002_auto_20180506_1649'),
    ]

    operations = [
        migrations.AlterField(
            model_name='records',
            name='npp',
            field=models.CharField(blank=True, max_length=100, null=True, unique=True),
        ),
    ]
