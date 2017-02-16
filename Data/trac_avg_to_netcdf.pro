; -----------------------------------------------------------
; NAME:
;       trac_avg_to_netcdf
;
; PURPOSE:
;       Turn trac_avg bitpunch data into netcdf files
;
; CALLING SEQUENCE:
;       trac_avg_to_netcdf
;
; NOTES:
;         Chartreuse is made from around 130 plants by monks confined to a monastary
;
; EXAMPLE:
;           trac_avg_to_netcdf, /DAILY
;               Reads a trac_avg month file which has daily output in it
;               creates a output.nc file
;
; CREATED:
;       In Edinburgh by Jesse 20170216
; UPDATED:
; -----------------------------------------------------------

pro trac_avg_to_netcdf, DAILY=DAILY
   
    ; SET UP FILE reading
    ;===============================================================
    path="/short/m19/jwg574/rundirs/geos5_2x25_tropchem/trac_avg/"
    gcfiles_pattern = path+'trac_avg.geos5*0000' ; should end with 0000
    gcfiles=file_search(gcfiles_pattern)
    print, gcfiles
    ; get pressure levels from geos5 2x25 model
    ; TODO: If possible get the pressures from the station file...
    g5=ctm_type('geos5_reduced',res=2)
    pmids=ctm_grid(g5,/pmid)
    
    ; Lets just do one file for now:
    gcfile=path+"trac_avg.geos5_2x25_tropchem.200501010000"

   
    ; END OF SETUP AREA
    ; ==============================================================
    
   
    ; In each file, grab metadata and pointers to array locations(di)
    ;
    ctm_get_data, di, diag, filename=gcfile, /QUIET
    if n_elements(di) eq 0 then print, "FILE DIDN'T LOAD ANYTHING"
    tnums=di[UNIQ(di.tracer, sort(di.tracer))].tracer
    ; n arrays to hold tracers
    ntracers=n_elements(tnums)
    tracer_arrs= ptrarr(ntracers)
        
    ; print tracer summary
    print, "Tracer names in these files:"
    print, di[UNIQ(di.tracername, sort(di.tracername))].tracername
    print, "Tracer numbers:"
    print, tnums
    
    ; go through tracers and save for ncdf
    taus = di[where(di.tracer eq tnums[0])].tau0
    ; Array sizes data
    ntaus=n_elements(taus)
    ntracers=n_elements(tnums)
    nlats=di[0].dim[1]
    nlons=di[0].dim[0]
    
    foreach tnum, tnums, tind do begin
        tracer=di[where(di.tracer eq tnum)]
        nlevels=tracer[0].dim[2]
        tracer_id[tind]= line.category+line.tracername
        ;tracer_category=make_array(ntracers, /string)
        tracer_data= make_array(ntaus, nlevels, nlats, nlons, ntracers, /double)
        tracer_unit= make_array(ntracers, /string)
        ; for each tracer we want, extract the data
        foreach tname, tnames, tindex do begin
            tlocs=where(stn_data.tracername eq tname)
            tracer_struct = stn_data[tlocs]
            tracer_unit[tindex] = stn_data[tlocs[0]].unit
            ; now pull out the data for this tracer
            for i=0,ntaus-1 do begin
                ctm_read_data, data_tmp, tracer_struct[i]
                tracer_data[i,*,tindex]=reform(data_tmp)
            endfor
        endforeach
        
        ; Free memory blocked by excessive use of GAMAP package
        ctm_cleanup
        
        ; build up our data arrays holding the tracer data
        if gcindex eq 0 then begin
            ; TAUS = array[times]
            taus_arr=taus
            ; DATA = array[times, levels]
            for i=0,ntracers-1 do begin
                tracer_arrs[i]=ptr_new(reform(tracer_data[*,*,i]))
            endfor
        endif else begin
            taus_arr=[taus_arr, taus]
            ; ADD Station File's tracers to overall tracer arrays
            for i=0,ntracers-1 do begin
                ; alldata = [alldata, filedata]
                *(tracer_arrs[i])=[*(tracer_arrs[i]),reform(tracer_data[*,*,i])]
            endfor
        endelse
           
        endforeach ; end of files loop
       
        ; save it all to NETCDF !!!!
        ;
       
        ; create netcdf file
        id=ncdf_create(outfilename,/clobber)
        ; set defaults for file
        ncdf_control, id, /fill
        ; define dimensions
        altid=ncdf_dimdef(id, 'altitude', nlevels)
        altid=ncdf_dimdef(id, 'latitude', nlats)
        altid=ncdf_dimdef(id, 'longitude', nlongs)
        ; unlimited dimension needs to be last!!!!!!!!!
        timeid=ncdf_dimdef(id, 'time', /unlimited)
        
        ; define variables
        tauid=ncdf_vardef(id, 'Tau', [timeid], /float)
        pressureid=ncdf_vardef(id,'Pressure', [altid], /float)
       
        ; attributes
        ncdf_attput, id, /GLOBAL, 'run', runstr
        ncdf_attput, id, /GLOBAL, 'resolution', resstr
        ncdf_attput, id, tauid, 'units','hours since 01 Jan 1985'
        ncdf_attput, id, pressureid, 'units', 'hPa'
       
        ;save each tracer
        ;
        dataid=intarr(ntracers)
        for i=0,ntracers-1 do begin
            dataid[i]=ncdf_vardef(id, tnames[i], [altid,timeid], /float)
            ncdf_attput, id, dataid[i], 'units', tracer_unit[i]
            ncdf_attput, id, dataid[i], 'long_name',tnames[i]
        endfor

        ; put file in data mode:
        ncdf_control, id, /endef

        ; input data
        ncdf_varput, id, tauid, taus_arr
        ncdf_varput, id, pressureid, pmids
        for i=0,ntracers-1 do begin
            ncdf_varput, id, dataid[i], transpose(*(tracer_arrs[i]))
        endfor
        ; close netcdf file
        ncdf_close, id
        print, "SAVED TO "+f_names[stn]
    endfor ; end of stations loop
end
