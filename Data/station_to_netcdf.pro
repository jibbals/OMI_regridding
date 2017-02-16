; -----------------------------------------------------------
; NAME:
;       station_to_netcdf
;
; PURPOSE:
;       Turn station data into netcdf files
;
; CALLING SEQUENCE:
;       station_to_netcdf
;
; NOTES:
;         Voles mate for life (I think it's because they don't live that long).
;
; EXAMPLE:
;           station_to_netcdf
;
; CREATED:
;       on top of a mountain by Jesse
;           13/4/2016 (ish)
; UPDATED:
;       Works for ANY chronological order of station and tracer inputs
;       Reads Units into netcdf maaaaaaannnnnnn
; -----------------------------------------------------------

pro station_to_netcdf
   
    ; SET UP FILE LOCATIONS AND STATION METADATA
    ;===============================================================
   
    ; Set filename and path for model data
    gcfiles_pattern = './stations.20*'
    ; each day needs to be looped over
    gcfiles=file_search(gcfiles_pattern)
    print, gcfiles
    ; get pressure levels from geos5 2x25 model
    ; TODO: If possible get the pressures from the station file...
    g5=ctm_type('geos5',res=2)
    pmids=ctm_grid(g5,/pmid)
       
    ; SET UP STATION ORDER AND METADATA
    ;
    snames=['Davis','Macquarie','Melbourne']
    s_names=snames
    f_names=s_names+'.nc'
    ;Melbourne 145E, 38S, Macquarie Island 159E, 55S, Davis 78E, 69S
    lats=['-68.58','-54.62','-37.81']
    lons=['77.97','158.86','144.96']
    siteI=[104, 137, 131 ]
    siteJ=[12, 19, 27]
    nlevels=72

   
    ; How many stations are we grabbing:
    nstations=n_elements(snames)
    tnames=['O3','PSURF','BXHEIGHT','AIRDEN']
    ntracers=n_elements(tnames)
   
    ; END OF SETUP AREA
    ; ==============================================================
    
    ; n arrays to hold tracers
    tracer_arrs= ptrarr(ntracers)
    printtracer=1
   
    ; loop through stations, for each station accumulate data then save to netcdf
    for stn=0, nstations-1 do begin
        ; this station has the following lon, lat index in input.geos
        sitelon=(siteI)[stn]
        sitelat=(siteJ)[stn]

        ; loop through files
        foreach gcfile, gcfiles, gcindex do begin
           
            ; In each file, grab metadata and pointers to array locations(di)
            ;
            ctm_get_data, di, diag, filename=gcfile, status=2, /QUIET
           
            ; print all tracers if we haven't yet:
            if printtracer eq 1 then begin
                print, "Tracer names in these files:"
                print, di[UNIQ(di.tracername, sort(di.tracername))].tracername
                print, "Tracers we will extract:"
                print, tnames
                printtracer=0
            endif
           
            ; all data related to this station can be pulled easily
            stn_data=di[where(di.first[0] eq sitelon and di.first[1] eq sitelat)]
           
            t0locs=where(stn_data.tracername eq tnames[0])
            taus = stn_data[t0locs].tau0
            ntaus=n_elements(taus)
            tracer_data= make_array(ntaus, nlevels, ntracers, /double)
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
        id=ncdf_create(f_names[stn],/clobber)
        ; set defaults for file
        ncdf_control, id, /fill
        ; define dimensions
        altid=ncdf_dimdef(id, 'altitude', nlevels)
        ; unlimited dimension needs to be last!!!!!!!!!
        timeid=ncdf_dimdef(id, 'time', /unlimited)
        ; define variables
        tauid=ncdf_vardef(id, 'Tau', [timeid], /float)
        pressureid=ncdf_vardef(id,'Pressure', [altid], /float)
       
        ; attributes
        ncdf_attput, id, /GLOBAL, 'station', s_names[stn]
        ncdf_attput, id, /GLOBAL, 'latitude', lats[stn]
        ncdf_attput, id, /GLOBAL, 'longitude', lons[stn]
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
