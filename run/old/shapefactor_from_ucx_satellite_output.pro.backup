;   Procedure name: SHAPEFACTOR_FROM_UCX_SATELLITE_OUTPUT
;
;   Porpoise:
;       Get shapefactor from satellite_output averaged at local time 1300-1400.
;       Save as .hdf5
;       Keep the sigma dimensions and create the shape factors used
;       in reformulating the AMF in OMHCHO data files.
;
;   Requirements: gamap2
;
;   History:
;       Created by Jesse Greenslade some time ago
;       Updated Fri-Oct 28: comments added and header (this bit)
;       Updated Thu-Dec 15: Altered for tropchem 47L
;       Updated Thu- 14/09/17: Returned to original format to calculate shape factors from UCX satellite output.
;           Renamed to shapefactor_from_ucx_satellite_output.pro
;           back to 72 levels
;           All lats/lons are again included
;       Updated Wed 20 sep 17: BUG FIXED: "psurf=ppbv/n_files" -> psurf=psurf/n_files
;

; Function to return linear or geometric midpoints from an array of pedges(or whatever)
function midpoints, pedges, geometric=geometric
    N = n_elements(pedges)
    inds=indgen(N-1)
    diffs=pedges[inds] - pedges[inds+1]
    mids= pedges[inds] + diffs/2.0
    if keyword_set(geometric) then $
        mids= sqrt(pedges[inds] * pedges[inds+1])

    return, mids
end

pro shapefactor_from_ucx_satellite_output, year, month

    ; first read all the files in the desired month
    yyyymm= string(year,format='(I4)') + string(month,format='(I02)')
    ptn='./ts_satellite.'+yyyymm+'*'
    files=file_search(ptn)
    
    ; use gamap2 function to grab grid
    ; subset determined from input.geos
    g5      = ctm_grid(ctm_type('GEOS5',res=2))
    lons    =g5.xmid;[117:137] ; not included in tropchem version..
    lats    =g5.ymid;[21:41]
    ;nlons=21
    ;nlats=21
    nlons   =g5.imx
    nlats   =g5.jmx
    
    ; arrays to hold monthly averages
    nlvls=72 ;47
    ppbv = dblarr(nlons,nlats, nlvls) ; molecules/1e9 molecules air
    psurf= dblarr(nlons,nlats, nlvls) ; hPa
    Nair = dblarr(nlons,nlats, nlvls) ; molecules air / cm3
    boxH = dblarr(nlons,nlats, nlvls) ; m
    
    ; loop through files
    foreach file, files, fi do begin
        print, 'reading ', file, '...'
        ; GC OUTPUT DIMENSIONS: 144, 91, nlvls)
        ; Read hcho ( molec/molec ) (ppbv in bitpunch notes)
        flag= CTM_GET_DATABLOCK( ppbvtmp, 'IJ-AVG-$', Tracer=20, File=file ) 
        ; Read psurf ( hPa )
        flag= CTM_GET_DATABLOCK( psurftmp, 'PEDGE-$', Tracer=1,  File=file )
        ; Read OH(loss?) ( molec/cm3 )
        ;flag= CTM_GET_DATABLOCK( ohlosstmp, 'CHEM-L=$', Tracer=1,  File=file )
        ; read number density of air( molec/cm3 )
        flag= CTM_GET_DATABLOCK( Nairtmp, 'TIME-SER', Tracer=7,  File=file )
        ; read the boxheights ( m )
        flag= CTM_GET_DATABLOCK( boxHtmp, 'BXHGHT-$', Tracer=1,  File=file )
        
        ppbv=ppbv+ppbvtmp
        psurf=psurf+psurftmp
        Nair=Nair+Nairtmp
        boxH=boxH+boxHtmp
    
    endforeach
    n_files=n_elements(files)
    ppbv=ppbv / n_files
    psurf=psurf / n_files
    Nair=Nair / n_files
    boxH=boxH / n_files
    
    ; pedge-psurf is bottom pressure for levels 1-nlvls
    pedges  = dblarr(nlons,nlats,nlvls+1)
    ; print, "mean surface pressure: ",mean(psurf[*,*,0] ; about 965 on 200501
    
    pedges[*,*,0:-2] = psurf
    pedges[*,*,-1] = 0.01 ; top of top edge is 0.01 hPa
    
    pmids   = dblarr(nlons,nlats,nlvls) ; hPa
    Nhcho   = dblarr(nlons,nlats,nlvls) ; will be molecules/m3
    S_z     = dblarr(nlons,nlats,nlvls) ; Shape Factor at nlvls altitudes
    S_sig   = dblarr(nlons,nlats,nlvls) ; shape factor at nlvls sigma levels
    sigma   = dblarr(nlons,nlats,nlvls) ; sigma levels
    
    ; timing
    t1=systime(1)
    
    ;use psurf to get pressure edges, then find geometric midpoints
    for x=0, nlons-1 do begin
        for y=0,nlats-1 do begin
            ; Get geometric pressure midpoints
            pmids[x,y,*] = midpoints(pedges[x,y,*],/geometric)
        endfor
    endfor
    
    ; Air density in molecules/cm3 -> molecules/m3
    Nair = double(Nair) * 1d6
    ; Total column AIR = vertical sum of (Nair * height)
    ; molecs/m2      = density(molecs/m3) * height(m)
    Omega_A = total( Nair * boxH , 3 )
    
    ; Density of HCHO in molecs/m3
    Nhcho   = double(ppbv)*double(Nair)*1d-9 ; molecs / m3
    
    ; TOTAL COLUMN HCHO = vertical sum of ( Nhcho * height)
    Omega_H   = total(Nhcho * boxH, 3) 
    
    ; Using Sigma coordinate: (1 at surface to 0 at toa)
    ; sigma = (P - P_t) / ( P_s - P_t )
    P_toa = pedges[*,*,-1]
    P_ttb = psurf[*,*,0] - P_toa ; pressure difference from surface to TOA
    for zz=0,nlvls-1 do begin
        Sigma[*,*,zz] = (pmids[*,*,zz] - P_toa) / P_ttb
    endfor
    
    ; normalized shape factors ( by definition from Palmer 01 )
    Omega_ratio = Omega_A / Omega_H  ; molecules Air/molecules HCHO
    mixingratio= ppbv * 1d-9 ; molecules HCHO/molecules AIR
    for ii=0, nlvls-1 do begin
        S_z[*,*,ii]  = Nhcho[*,*,ii] / Omega_H ; 1/m
        S_sig[*,*,ii] = Omega_ratio * mixingratio[*,*,ii] ; dimensionless
    endfor
    
    ; VCHCHO: molecs/m2, number densities: molecs/m3
    structdata={VCHCHO:Omega_H, NHCHO:Nhcho, ShapeZ:S_z, $
                VCAir:Omega_A, NAir:Nair, ShapeSigma:S_sig, $
                latitude:lats, longitude:lons, PEdges:pedges, PMids:pmids, $
                Sigma:Sigma, boxheights:boxH}

    ; write a structure to hdf5 as a compound datatype
    fout = './ucx_shapefactor_'+yyyymm+'.he5'
    fid = H5F_CREATE(fout)
    
    datatype_id = H5T_IDL_CREATE(structdata)
    dataspace_id = H5S_CREATE_SIMPLE(1) ; not so simple..
    
    ; name of compound datatype
    dataset_id = H5D_CREATE(fid,'GC_UCX_HCHOColumns',datatype_id,dataspace_id)
    H5D_WRITE, dataset_id, structdata
    
    H5S_CLOSE, dataspace_id
    H5T_CLOSE, datatype_id

    H5F_CLOSE, fid
    print, 'Wrote file to ', fout
end
