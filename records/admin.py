# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.contrib import admin

import csv
from django.http import HttpResponse

# Register your models here.
from .models import Records, Presensi

class PresensiAdmin(admin.ModelAdmin):
    list_display = ("tanggal","pegawai","jam_masuk","jam_pulang","kehadiran")
    list_filter = ("tanggal", "pegawai","kehadiran")
    actions = ["export_as_csv"]
    # def export_as_csv(self, request, queryset):
    #     meta = self.model._meta
    #     field_names = [field.name for field in meta.fields]
    #
    #     response = HttpResponse(content_type='text/csv')
    #     response['Content-Disposition'] = 'attachment; filename={}.csv'.format(meta)
    #     writer = csv.writer(response)
    #
    #     writer.writerow(field_names)
    #     for obj in queryset:
    #         row = writer.writerow([getattr(obj, field) for field in field_names])
    #
    #     return response
    def export_as_csv(self, request, queryset):
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=Report_Absensi.csv'
        writer = csv.writer(response)

        writer.writerow(['Tanggal','NPP','Pegawai','Jam Masuk','Jam Pulang','Kehadiran'])
        for obj in queryset:
            kehadiran = '-'
            if obj.kehadiran == 0:
                kehadiran = 'Alpha'
            elif obj.kehadiran == 1:
                kehadiran = 'Hadir'
            elif obj.kehadiran == 2:
                kehadiran = 'Izin'
            elif obj.kehadiran == 3:
                kehadiran = 'Sakit'
            row = writer.writerow([obj.tanggal,obj.pegawai.npp,obj.pegawai,obj.jam_masuk,obj.jam_pulang,kehadiran])

        return response

    export_as_csv.short_description = "Export Selected"

admin.site.register(Records)
admin.site.register(Presensi, PresensiAdmin)