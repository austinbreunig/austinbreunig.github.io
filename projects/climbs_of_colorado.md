---
layout: post
title: 'Climbs of Colorado: Web-scraping Mountain Project'
---
### Introduction
Colorado has an abundance of rock climbing. By web scraping Mountain Project data, I was able to aggregate all of Colorado's climbing onto a single map. The following notebook searches through various key text found on the site and pulls location data and number of routes per type of climb for each crag. A buffer is conducted on each point (converts to a polygon) to then utilize the "height" feature in Keplr.gl. Height can then be defined by any numerical value to visualize the data. 

The final map can be found [here](https://kepler.gl/demo/map?mapUrl=https://dl.dropboxusercontent.com/s/9jk7st9u21mqtcf/keplergl_47u2wdf.json)

### Notebook
```python
import os
os.environ['USE_PYGEOS'] = '0'
from bs4 import BeautifulSoup
import requests
import pandas as pd
from ast import literal_eval
import geopandas as gpd

```


```python
# Main Landing Page for all Colorado Climbing on Mountain Project
CO_URL = 'https://www.mountainproject.com/area/105708956/colorado'
```


```python
# Get request and parse using Beuatiful Soup
html_str = requests.get(CO_URL).content
soup = BeautifulSoup(html_str, 'html.parser')
```

By looping through the 'lef-nav-row' class and searching for text within the 'a' tag, the name of the area along with its url can be found. 
Elements are added to a dictionary called 'co_areas'


```python
co_areas = {}
for link in soup.find_all('div', {'class': 'lef-nav-row'}):
    for url in link.find_all('a', href=True):
        print(url.contents)
        print(url['href'])
        co_areas[url.contents[0]] = url['href']
```

```python
co_areas
```

    {'10 Mile Canyon': 'https://www.mountainproject.com/area/106817352/10-mile-canyon',
     'Alpine Rock': 'https://www.mountainproject.com/area/105744466/alpine-rock',
     'Aspen Glen Campground Bouldering': 'https://www.mountainproject.com/area/122921638/aspen-glen-campground-bouldering',
     'Black Hawk': 'https://www.mountainproject.com/area/121624024/black-hawk',
     'Boulder': 'https://www.mountainproject.com/area/105801420/boulder',
     'Breckenridge': 'https://www.mountainproject.com/area/108289116/breckenridge',
     'Bridge': 'https://www.mountainproject.com/area/124014835/bridge',
     'Broomfield': 'https://www.mountainproject.com/area/122059772/broomfield',
     'Broomfield Boulders': 'https://www.mountainproject.com/area/123696872/broomfield-boulders',
     'Buena Vista': 'https://www.mountainproject.com/area/105744391/buena-vista',
     'Canon City': 'https://www.mountainproject.com/area/105800427/canon-city',
     'Carbondale Area': 'https://www.mountainproject.com/area/105802064/carbondale-area',
     'Cataract Lake (Heeney, Colorado)': 'https://www.mountainproject.com/area/106562735/cataract-lake-heeney-colorado',
     'CO Ice & Mixed': 'https://www.mountainproject.com/area/105807296/co-ice-mixed',
     'Coal Creek Canyon': 'https://www.mountainproject.com/area/108580517/coal-creek-canyon',
     'Colorado Springs': 'https://www.mountainproject.com/area/105800307/colorado-springs',
     'Craig/Hamilton Area': 'https://www.mountainproject.com/area/123438738/craighamilton-area',
     'Crested Butte': 'https://www.mountainproject.com/area/107163609/crested-butte',
     'Crestone': 'https://www.mountainproject.com/area/118890747/crestone',
     'Cross Mountain Boulders': 'https://www.mountainproject.com/area/113745735/cross-mountain-boulders',
     'Denver Metropolitan Area Bouldering and Buildering': 'https://www.mountainproject.com/area/121107762/denver-metropolitan-area-bouldering-and-buildering'}




```python
# Create new dataframe from dictionary
colorado_areas = pd.DataFrame.from_dict(co_areas, orient='index')
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10 Mile Canyon</th>
      <td>https://www.mountainproject.com/area/106817352...</td>
    </tr>
    <tr>
      <th>Alpine Rock</th>
      <td>https://www.mountainproject.com/area/105744466...</td>
    </tr>
    <tr>
      <th>Aspen Glen Campground Bouldering</th>
      <td>https://www.mountainproject.com/area/122921638...</td>
    </tr>
    <tr>
      <th>Black Hawk</th>
      <td>https://www.mountainproject.com/area/121624024...</td>
    </tr>
    <tr>
      <th>Boulder</th>
      <td>https://www.mountainproject.com/area/105801420...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>Telluride/Norwood area</th>
      <td>https://www.mountainproject.com/area/105969849...</td>
    </tr>
    <tr>
      <th>Vogel Canyon</th>
      <td>https://www.mountainproject.com/area/107841192...</td>
    </tr>
    <tr>
      <th>Weston Pass-(Weston Wall)</th>
      <td>https://www.mountainproject.com/area/106689909...</td>
    </tr>
    <tr>
      <th>Wet Mountains, The</th>
      <td>https://www.mountainproject.com/area/105801976...</td>
    </tr>
    <tr>
      <th>Wild Mountain</th>
      <td>https://www.mountainproject.com/area/119270576...</td>
    </tr>
  </tbody>
</table>
<p>75 rows × 1 columns</p>
</div>




```python
# Function takes coordinate tag from html and parses lat and long for it
def get_coords(contents_list):
    coord_str = contents_list[0].strip() # remove linebreak/whitespace
    coord_list = coord_str.split(',')
    lat = float(coord_list[0])
    lng = float(coord_list[1])

    return lat, lng
```

The following block loops through each area and does a get request to pull crag information. If an area has sub-areas, then another get requests drills further to obtain crag info. Once it has reached the crag-level, information is pull such as lat/long and climbing stats. Each crag is saved to a dictionary which will later make up the final dataframe. 


```python
all_crags = pd.DataFrame()
for area, url in colorado_areas.iterrows():
    print(area)
    crag_coords = {}
    html_str = requests.get(url[0]).content
    area_soup = BeautifulSoup(html_str, 'html.parser')
    
    for link in area_soup.find_all('div', {'class': 'lef-nav-row'}):
        for url in link.find_all('a', href=True):
            crag_name = url.contents[0]

            html_str = requests.get(url['href']).content
            crag = BeautifulSoup(html_str, 'html.parser')
            
            for c in crag.find_all('div', {'class': 'col-md-3 left-nav float-md-left mb-2'}):
                for header in c.find_all('h3'):
                    if 'Areas in' in header.text:
                        print(f'Has subareas: {header.text}')

                        # get subarea names
                        for link in crag.find_all('div', {'class': 'lef-nav-row'}):
                            for url in link.find_all('a', href=True):
                                sub_crag = url.contents[0]
                                print(crag_name)
                                html_str = requests.get(url['href']).content
                                subarea_soup = BeautifulSoup(html_str, 'html.parser')

                                for c in subarea_soup.find_all('div', {'class': 'small'}):
                                    # Grabbing lat + long
                                    for text in c.find_all('td'):
                                        if '-1' in text.text:
                                            contents_list = text.contents
                                            lat, lng = get_coords(contents_list)
                                            #print(lat, lng)
                                            crag_coords = {'Lat': lat, 'Long': lng}
                                            #print(crag_coords)


                                # total climbs stats
                                for c in subarea_soup.find_all('div', {'class': 'col-lg-6 text-xs-center', 'id': 'route-count-container'}):
                                    script_tag = c.find_all('script')[1].string
                                    if script_tag:
                                        tag = script_tag.split(';')[0].replace(' ', '').replace('\n', '')
                                        data = tag.split('google.visualization.arrayToDataTable(')[1].rstrip(')')
                                        data = literal_eval(data)[1:]
                                        climb_dict = {k[0]: k[1] for k in data}
                                        crag_dict = climb_dict | crag_coords
                                        crag_id = pd.MultiIndex.from_tuples([(crag_name, sub_crag)], name=['Area', 'Crag'])
                                        crag_df = pd.DataFrame(crag_dict, index=crag_id)
                                        all_crags = pd.concat([all_crags, crag_df])


                    else:
                        print(f'No subareas: {header.text}')
                        
                        for c in crag.find_all('div', {'class': 'small'}):
                            # Grabbing lat + long
                            for text in c.find_all('td'):
                                if '-1' in text.text:
                                    contents_list = text.contents
                                    lat, lng = get_coords(contents_list)
                                    #print(lat, lng)
                                    crag_coords = {'Lat': lat, 'Long': lng}
                                    #print(crag_coords)


                        # total climbs stats
                        for c in crag.find_all('div', {'class': 'col-lg-6 text-xs-center', 'id': 'route-count-container'}):
                            script_tag = c.find_all('script')[1].string
                            if script_tag:
                                tag = script_tag.split(';')[0].replace(' ', '').replace('\n', '')
                                data = tag.split('google.visualization.arrayToDataTable(')[1].rstrip(')')
                                data = literal_eval(data)[1:]
                                climb_dict = {k[0]: k[1] for k in data}
                                crag_dict = climb_dict | crag_coords
                                crag_id = pd.MultiIndex.from_tuples([(area, crag_name)], name=['Area', 'Crag'])
                                crag_df = pd.DataFrame(crag_dict, index=crag_id)
                                all_crags = pd.concat([all_crags, crag_df])
                                
                                #all_crag_ids.append(crag_tuple)
                  
The all_crags geodataframe consist of each areas crags (as Multi-level Index), along with the number of climbs for each discipline. The lat long has been converted to a Point geometry.


```python
# Fill in nulls with 0s.
all_crags = all_crags.fillna(0)
all_crags.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Sport</th>
      <th>Lat</th>
      <th>Long</th>
      <th>Trad</th>
      <th>Toprope</th>
      <th>Boulder</th>
      <th>Alpine</th>
      <th>Aid</th>
      <th>Ice</th>
      <th>Mixed</th>
      <th>Total Climbs</th>
      <th>geometry</th>
    </tr>
    <tr>
      <th>Area</th>
      <th>Crag</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10 Mile Canyon</th>
      <th>10 Mile Tower</th>
      <td>1.0</td>
      <td>39.57010</td>
      <td>-106.11985</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>POINT (-106.11985 39.57010)</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Brick Wall/Alcoves</th>
      <th>Alcoves, The</th>
      <td>0.0</td>
      <td>39.56660</td>
      <td>-106.12615</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>POINT (-106.12615 39.56660)</td>
    </tr>
    <tr>
      <th>Brick Wall</th>
      <td>0.0</td>
      <td>39.56637</td>
      <td>-106.12630</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>POINT (-106.12630 39.56637)</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">10 Mile Canyon</th>
      <th>Cache Wall, The</th>
      <td>6.0</td>
      <td>39.56147</td>
      <td>-106.13202</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>POINT (-106.13202 39.56147)</td>
    </tr>
    <tr>
      <th>Desperation Wall</th>
      <td>4.0</td>
      <td>39.51999</td>
      <td>-106.14030</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>POINT (-106.14030 39.51999)</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a Total Climbs field which represents total climbs per crag.
all_crags['Total Climbs'] = all_crags.apply(lambda x: x['Sport'] + x['Trad'] + x['Toprope'] + x['Boulder'] + x['Alpine'] + x['Ice'] + x['Aid'] + x['Mixed'], axis=1)
```


```python
# Convert lat/long to Point geometry
gdf = gpd.GeoDataFrame(all_crags, geometry=gpd.points_from_xy(all_crags['Long'], all_crags['Lat']), crs=4326)
gdf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Sport</th>
      <th>Lat</th>
      <th>Long</th>
      <th>Trad</th>
      <th>Toprope</th>
      <th>Boulder</th>
      <th>Alpine</th>
      <th>Aid</th>
      <th>Ice</th>
      <th>Mixed</th>
      <th>Total Climbs</th>
      <th>geometry</th>
    </tr>
    <tr>
      <th>Area</th>
      <th>Crag</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10 Mile Canyon</th>
      <th>10 Mile Tower</th>
      <td>1.0</td>
      <td>39.57010</td>
      <td>-106.11985</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>POINT (-106.11985 39.57010)</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Brick Wall/Alcoves</th>
      <th>Alcoves, The</th>
      <td>0.0</td>
      <td>39.56660</td>
      <td>-106.12615</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>POINT (-106.12615 39.56660)</td>
    </tr>
    <tr>
      <th>Brick Wall</th>
      <td>0.0</td>
      <td>39.56637</td>
      <td>-106.12630</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>POINT (-106.12630 39.56637)</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">10 Mile Canyon</th>
      <th>Cache Wall, The</th>
      <td>6.0</td>
      <td>39.56147</td>
      <td>-106.13202</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>POINT (-106.13202 39.56147)</td>
    </tr>
    <tr>
      <th>Desperation Wall</th>
      <td>4.0</td>
      <td>39.51999</td>
      <td>-106.14030</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>POINT (-106.14030 39.51999)</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Keplr gl maps only work in the projection Web Mercator (3857)
gdf = gdf.to_crs(epsg=3857)
gdf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Sport</th>
      <th>Lat</th>
      <th>Long</th>
      <th>Trad</th>
      <th>Toprope</th>
      <th>Boulder</th>
      <th>Alpine</th>
      <th>Aid</th>
      <th>Ice</th>
      <th>Mixed</th>
      <th>Total Climbs</th>
      <th>geometry</th>
    </tr>
    <tr>
      <th>Area</th>
      <th>Crag</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10 Mile Canyon</th>
      <th>10 Mile Tower</th>
      <td>1.0</td>
      <td>39.57010</td>
      <td>-106.11985</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>POINT (-11813207.665 4803665.639)</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Brick Wall/Alcoves</th>
      <th>Alcoves, The</th>
      <td>0.0</td>
      <td>39.56660</td>
      <td>-106.12615</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>POINT (-11813908.978 4803160.209)</td>
    </tr>
    <tr>
      <th>Brick Wall</th>
      <td>0.0</td>
      <td>39.56637</td>
      <td>-106.12630</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>POINT (-11813925.676 4803126.996)</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">10 Mile Canyon</th>
      <th>Cache Wall, The</th>
      <td>6.0</td>
      <td>39.56147</td>
      <td>-106.13202</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>POINT (-11814562.423 4802419.439)</td>
    </tr>
    <tr>
      <th>Desperation Wall</th>
      <td>4.0</td>
      <td>39.51999</td>
      <td>-106.14030</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>POINT (-11815484.149 4796431.759)</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Dataframe is buffered to create polygon geometry which can be represented by height in the Keplr visualization.
# 100-meters was used to create a large enough polygon without loosing locational accuracy.
gdf_buffer = gpd.GeoDataFrame(gdf.drop('geometry', axis=1), geometry=gdf.buffer(100, cap_style=3), crs=3857)

```


```python
# Write to geojson
gdf_buffer.to_file('all_crags_3857_10m.geojson', driver='GeoJSON')
```
