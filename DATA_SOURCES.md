# Data sources

- AIS replay window: [NOAA MarineCadastre AIS](https://marinecadastre.gov/ais/)
- Monterey Bay sanctuary boundary: [NOAA National Marine Sanctuaries GIS](https://sanctuaries.noaa.gov/library/imast_gis.html)
- Coastline geometry: [Natural Earth](https://www.naturalearthdata.com/)
- Sanctions demonstration subset: [U.S. Treasury OFAC Sanctions List](https://ofac.treasury.gov/sanctions-list-service)
- Submarine cable demonstration geometry: [TeleGeography Submarine Cable Map](https://www.submarinecablemap.com/)
- Basemap tiles: [CARTO](https://carto.com/attributions) with
  [OpenStreetMap](https://www.openstreetmap.org/copyright) data
- Navigable-water constraints: [OpenFreeMap](https://openfreemap.org/) vector
  tiles using the [OpenMapTiles](https://openmaptiles.org/schema/) schema and
  [OpenStreetMap](https://www.openstreetmap.org/copyright) coastline, harbor,
  canal, river, lake, and ocean geometry. Tiles are cached locally after use.
- Optional live positions: [AISStream](https://aisstream.io/)
- Port identity and facilities: [NGA World Port Index](https://msi.nga.mil/Publications/WPI)
  via its public ArcGIS FeatureServer
- Optional vessel identity, fishing/encounter/loitering/port-visit/AIS-gap
  events, apparent fishing effort, and SAR vessel detections:
  [Global Fishing Watch APIs](https://globalfishingwatch.org/our-apis/).
  Fishing events and effort are modelled from AIS; SAR detections come from
  Copernicus Sentinel-1 analysis. Neither is proof of illegal activity.
- Marine protected-area and OECM attributes: Protected Planet
  WDPA/WDOECM Public marine CSV, August 2026. The supplied CSV has no geometry,
  so Aegis uses it to name and describe protected areas referenced by Global
  Fishing Watch events rather than drawing invented boundaries.
- EEZ reference and license: Flanders Marine Institute (2023), Maritime
  Boundaries Geodatabase version 12, DOI
  [10.14284/632](https://doi.org/10.14284/632). The supplied folder contains
  the license but not the `.gpkg`; Aegis therefore does not draw those legal
  boundaries from the incomplete download.
- NGA Pub. 150 World Port Index, 27th edition (2019), is retained as an
  archival reference. Live port context continues to use the newer NGA
  FeatureServer instead of replacing current records with the 2019 edition.
- Surface-current analysis and forecast:
  [Copernicus Marine Service](https://marine.copernicus.eu/)
- Optional bathymetry overlay: GEBCO Bathymetric Compilation Group 2026,
  [GEBCO_2026 Grid](https://www.gebco.net/data-products-gridded-bathymetry-data/gebco2026-grid)
- Response-planning resource rates: U.S. Coast Guard National Pollution Funds
  Center Electronic CG-5136 Cost Documentation Workbook, FY26 Reimbursable
  Standard Rates effective October 1, 2025,
  [Cost Documentation](https://www.uscg.mil/Mariners/National-Pollution-Funds-Center/Documentation-Cost/)

Scenario and sanctions records are demonstration data, not live intelligence.
GEBCO data are contextual and must not be used for navigation or safety at sea.
CG-5136 figures are outside-government reimbursable rates, not incurred costs,
damages, dispatch recommendations, or inside-government accounting rates.
Protected-area status does not mean every form of fishing is prohibited.
Rules vary by designation, zone, species, gear, season, and authorization.
Consult each provider's current terms before redistributing or deploying the
datasets outside this project.
