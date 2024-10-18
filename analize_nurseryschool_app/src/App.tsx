import React, { useState, useEffect } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import { LatLng } from 'leaflet';
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import Papa from "papaparse";

// Interface for CSV data
interface FacilityData {
  facilityName: string;
  analize_service: string;
  //teacher: string;
  //startTime: string;
  //endTime: string;
  //capacity: string;
  lon: number;
  lat: number;
  feature_color: string;
  img_base64: string;
}

// Icon customization
const createIconWithRGB = (color: string) => {
  return L.divIcon({
    className: 'custom-icon',  // アイコンのCSSクラス
    html: `<div style="background-color: ${color}; width: 24px; height: 24px; border-radius: 50%;"></div>`,
    iconSize: [24, 24],
    iconAnchor: [12, 12],
    popupAnchor: [1, -34],
    shadowSize: [41, 41],
  });
};

const App = () => {
  const [data, setData] = useState<FacilityData[]>([]);
  const position = new LatLng(35.717, 139.747)
  //const position = new LatLng(35.675, 139.659)

  // 非同期関数でCSVを取得
  const fetchCSV = async () => {
    try {
      // CSVファイルをフェッチ
      const response = await fetch('/nursery_data2.csv');
      //const response = await fetch('/unlicensed_nursery_data.csv');
      
      if (!response.ok) {
        throw new Error('CSVの取得に失敗しました。ステータスコード: ' + response.status);
      }

      // テキストとして読み込み
      const csvText = await response.text();
      console.log(csvText)

      // CSVデータを解析
      Papa.parse(csvText, {
        header: true,
        complete: (result) => {
          const parsedData = result.data as FacilityData[];

          // 緯度経度のデータが有効かどうかをチェック
          const validData = parsedData.filter((facility) =>
            facility.lat !== undefined && facility.lon !== undefined &&
            !isNaN(facility.lat) && !isNaN(facility.lon)
          );

          setData(validData);  // 有効なデータをセット
        },
      });
    } catch (err) {
      console.error('Error fetching or parsing CSV:', error)
    }
  };

  useEffect(() => {
    fetchCSV();  // CSV取得関数を呼び出し
  }, []);

  return (
    <React.Fragment>
      <h1>Nursery school evaluation and analysis system for professionals</h1>
      <details>
        <summary>Overview</summary>
        The aim is to provide a support system for education professionals and local authorities when analysing nursery schools and kindergartens.
      </details>
      
      <label class="toggleContainer">
        <span class="toggleLabel">Compare two nursery schools</span>
        <label class="toggleButton">
          <input type="checkbox" class="toggleButton__checkbox"/>
        </label>
      </label>

      <MapContainer center={position} zoom={15} style={{ height: "800px", width: "1200px" }}>
        <TileLayer
          url="https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg"
          attribution="(5/2023)"
        />
       {data.map((facility, index) => (
          <Marker
            key={index}
            position={[facility.lat, facility.lon]}
            icon={createIconWithRGB(facility.feature_color)}
          >
            <Popup>
              <div>
                <h3>{facility.facilityName}</h3>
                <img src={`data:image/png;base64,${facility.img_base64}`} alt={facility.facilityName} width="300px" height="250px"/>
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </React.Fragment>
  );
};

export default App;
