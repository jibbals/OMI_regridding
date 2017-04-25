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

pro trac_avg_to_netcdf, infilename=infilename, outfilename=outfilename
    ; SET UP FILE reading
    ;===============================================================
    path="/short/m19/jwg574/rundirs/geos5_2x25_tropchem/trac_avg/"
    ;gcfiles_pattern = path+'trac_avg.geos5*0000' ; should end with 0000
    ;gcfiles=file_search(gcfiles_pattern)
    ;print, gcfiles
    runstr='GEOS5 2X25_47 Tropchem TomsUpdated'
    resstr='2 lat 2.5 long 47 levels'
    ; Turn off plot displays
    set_plot, 'NULL'
    ; get dimensions for our model run
    nlevels=47
    g5=ctm_type('geos5',nlayers=nlevels,res=2)
    pmids=ctm_grid(g5,/pmid)
    lats=ctm_grid(g5,/ymid)
    lons=ctm_grid(g5,/xmid)
    
    ; Lets just do one file for now:
        ;gcfile=path+"trac_avg.geos5_2x25_tropchem.200501010000" $
    if n_elements(infilename) eq 0 then $
        gcfile=path+"trac_avg.geos5_2x25_tropchem.200501010000" $
    else gcfile=infilename
    
    if n_elements(outfilename) eq 0 then outfilename=gcfile+'.nc'
    
    ; END OF SETUP AREA
    ; ==============================================================
    
   
    ; In each file, grab metadata and pointers to array locations(di)
    ;
    ctm_get_data, di, diag, filename=gcfile, /QUIET
    if n_elements(di) eq 0 then print, "FILE DIDN'T LOAD ANYTHING"
    tnums=di[UNIQ(di.tracer, sort(di.tracer))].tracer
    ; n arraysV to hold tracers
        
    ; print tracer summary
    print, "Tracer names in these files:"
    print, di[UNIQ(di.tracername, sort(di.tracername))].tracername
    print, n_elements(tnums), " Tracer numbers in file"
    
    ; anthsrc and biogsrc categories have matching tnums!
    taus = di[where(di.tracer eq tnums[0])].tau0
    id_uniq= di.category+string(di.tracer)
    id_uniq= UNIQ(id_uniq, sort(id_uniq))
    ntracers=n_elements(id_uniq) ; number of outputs to save
    print, ntracers, " GC outputs to be recorded"
    
    tracer_arrs =   ptrarr(ntracers)
    taus_arrs   =   ptrarr(ntracers)
    tracer_unit =   make_array(ntracers,/string)
    ;tracer_name =   make_array(ntracers,/string)
    tracer_dims =   ptrarr(ntracers)
    tracer_id   =   make_array(ntracers,/string)
    ; Array sizes data
    ntaus=n_elements(taus)
    nlats=di[0].dim[1]
    nlons=di[0].dim[0]
    
    ; For every tracer -> grab all data
    ; go through tracers and save for ncdf
    foreach uniq_ind, id_uniq, counter do begin
        tnum=di[uniq_ind].tracer
        category=di[uniq_ind].category
        tracer=di[where(di.tracer eq tnum and di.category eq category)]
        trac0=tracer[0]
        tdims=trac0.dim[where(trac0.dim gt 0)] ; zero dimensions are no good for make_array
        ;tracer_name[counter]=trac0.tracername
        tracer_id[counter]= trac0.category+trac0.tracername
        tracer_taus=n_elements(tracer.tau0)
        if tracer_taus ne ntaus then begin
            print, 'ntaus=',ntaus,' while this tracer has ', tracer_taus, ' taus:'
            print, tracer.tau0
        endif
        
        dims=[tracer_taus, tdims]
        print, tnum, ' ' + tracer_id[counter], dims
        
        tracer_data= make_array(DIMENSION=dims, /double)
        ; for each tracer we want, extract the data
        tracer_unit[counter] = trac0.unit
        ; now pull out the data for this tracer
        for i=0,tracer_taus-1 do begin
            ctm_read_data, data_tmp, tracer[i]
            tracer_data[i,*,*,*]=data_tmp;reform(data_tmp)
        endfor
        
        ; build up our data arrays holding the tracer data
        tracer_dims[counter]=ptr_new(dims)
        tracer_arrs[counter]=ptr_new(tracer_data) ; save ref to array of data before moving on
        ;stop
    endforeach
    
    ; Free memory blocked by excessive use of GAMAP package
    ctm_cleanup
       
    ; save it all to NETCDF !!!!
    ;
    
    ; create netcdf file
    id=ncdf_create(outfilename,/clobber)
    ; set defaults for file
    ncdf_control, id, /fill
    ; define dimensions
    altid=ncdf_dimdef(id, 'altitude', nlevels)
    latid=ncdf_dimdef(id, 'latitude', nlats)
    lonid=ncdf_dimdef(id, 'longitude', nlons)
    ; unlimited dimension needs to be last!!!!!!!!!
    timeid=ncdf_dimdef(id, 'time', ntaus);/unlimited)
    
    ; define variables
    tauid=ncdf_vardef(id, 'Tau', [timeid], /float)
    pressureid=ncdf_vardef(id,'Pressure', [altid], /float)
    latvarid=ncdf_vardef(id, 'latitude', [latid], /float)
    lonvarid=ncdf_vardef(id, 'longitude', [lonid], /float)

    ; attributes
    
    ncdf_attput, id, /GLOBAL, 'run', runstr
    ncdf_attput, id, /GLOBAL, 'resolution', resstr
    ncdf_attput, id, tauid, 'units','hours since 01 Jan 1985'
    ncdf_attput, id, pressureid, 'units', 'hPa'
    ncdf_attput, id, pressureid, 'description', 'pressure midpoints'
    ncdf_attput, id, latvarid, 'units', 'degrees north'
    ncdf_attput, id, latvarid, 'description', 'gridbox midpoints'
    ncdf_attput, id, lonvarid, 'units', 'degrees east'
    ncdf_attput, id, lonvarid, 'description', 'gridbox midpoints'
    
    ;save each tracer
    ;
    print, "Saving data to netcdf"
    dataid=intarr(ntracers)
    dim_ids=[timeid,lonid,latid,altid]
    for i=0,ntracers-1 do begin
        print, tracer_id[i], ' dimensions:',*(tracer_dims[i])
        ; Only use dimensions with more than 1 length
        tracer_dim_id=dim_ids[where((*tracer_dims[i]) gt 1)]
        dataid[i]=ncdf_vardef(id, tracer_id[i], tracer_dim_id, /float)
        ncdf_attput, id, dataid[i], 'units', tracer_unit[i]
        ncdf_attput, id, dataid[i], 'long_name',tracer_id[i]
    endfor

    ; put file in data mode:
    ncdf_control, id, /endef

    ; input data
    ncdf_varput, id, tauid, taus
    ncdf_varput, id, pressureid, pmids
    ncdf_varput, id, latvarid, lats
    ncdf_varput, id, lonvarid, lons
    
    for i=0,ntracers-1 do begin
        ; ncdf_varput uses normal matrix subscript order - idl uses transpose
        ;ncdf_varput, id, dataid[i], transpose(*(tracer_arrs[i]))
        help, reform(*tracer_arrs[i])
        ncdf_varput, id, dataid[i], reform(*(tracer_arrs[i]))
    endfor
    ; close netcdf file
    ncdf_close, id
    print, "SAVED TO "+outfilename
    ;endfor ; end of stations loop
end
