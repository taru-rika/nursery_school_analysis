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
  const [selectedColor, setSelectedColor] = useState<string | null>(null);
  const [selectedFacility, setSelectedFacility] = useState(null);

  const handleMarkerClick = (facility) => {
    setSelectedFacility(facility);
    console.log(facility)
  };

  //const position = new LatLng(35.675, 139.659)

  const colors = ["#9e0142", "#ffffbf", "#f88d52", "#89d0a4", "#5e4fa2"]
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
    <div>
    <h1>Nursery school evaluation and analysis system for Educational Professionals</h1>
     <details>
       <summary>Overview</summary>
       The aim is to provide a support system for education professionals and local authorities when analysing nursery schools and kindergartens.
     </details>
     <h2>Please Click some Nursery Schools!</h2>
     <div>
       {colors.map(color => (
         <button key={color} onClick={() => setSelectedColor(color)}>
           Showing {color} pins
         </button>
       ))}
       <button onClick={() => setSelectedColor(null)}>Showing All pins</button>
     </div> 
     
     <div className='app-container'>
       <div className="map-container">
        <MapContainer center={position} zoom={15} style={{ height: "800px", width: "100%" }}>
          <TileLayer
            url="https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg"
            attribution="(5/2023)"
          />
         {data
           .filter(pin => !selectedColor || pin.feature_color === selectedColor) 
           .map((facility, index) => (
            <Marker
              key={index}
              position={[facility.lat, facility.lon]}
              icon={createIconWithRGB(facility.feature_color)}
              eventHandlers={{
                click: () => handleMarkerClick(facility),
              }}
            />
           ))}
        </MapContainer>
       </div>
       {selectedFacility && (
         <div className="info-container">
           <div>
             <h2 style={{ display: 'inline-block', textAlign: 'left' }}>{selectedFacility.facilityName}</h2>
             <button
               onClick={() => setSelectedFacility(null)}  // ボタンをクリックで表示を閉じる
               style={{
                 display: 'inline-block',
                 backgroundColor: 'transparent',
                 border: 'none',
                 fontSize: '20px',
                 cursor: 'pointer'
               }}
             >
               &times;
             </button>
             <iframe
               className='network-html'
               src={`cooccurrence_network_${selectedFacility.facilityName}.html`}
               style={{ width: "100%", height: "800px" }}
             ></iframe>
           </div>
         </div>
       )}
     </div>
   </div>
  );
};

export default App;
