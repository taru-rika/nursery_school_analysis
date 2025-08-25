// main.js（Reactを使わない純粋なJS版）
import "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js";

const map = L.map("map").setView([35.717, 139.747], 14);

L.tileLayer("https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg", {
  attribution: "(5/2023)",
}).addTo(map);
setTimeout(() => {
  map.invalidateSize();
}, 300);

const colors = ["#9e0142", "#ffffbf", "#f88d52", "#89d0a4", "#5e4fa2"];
let selectedColor = null;
let selectedFacility = null;
let selectedFacilities = [];
let compareMode = false;

function createIconWithRGB(color) {
  return L.divIcon({
    className: "custom-icon",
    html: `<div style="background-color: ${color}; width: 24px; height: 24px; border-radius: 50%;"></div>`
  });
}

let allFacilities = [];
function loadCSV() {
  fetch("/nursery_data2.csv")
    .then((response) => response.text())
    .then((csvText) => {
      Papa.parse(csvText, {
        header: true,
        complete: (result) => {
          const parsedData = result.data || [];
          const facilities = parsedData.filter(facility =>
            facility.lat && facility.lon && !isNaN(facility.lat) && !isNaN(facility.lon)
          );

          allFacilities = facilities;
          renderMarkers(allFacilities);
        },
      });
    });
}

function renderMarkers(facilities) {
  facilities.forEach((facility) => {
    if (!selectedColor || facility.feature_color === selectedColor) {
      const marker = L.marker([facility.lat, facility.lon], {
        icon: createIconWithRGB(facility.feature_color),
      }).addTo(map);

      marker.on("click", () => {
        if (compareMode) {
          selectedFacility = facility;
          if (!selectedFacilities.find(f => f.facilityName === facility.facilityName)) {
            if (selectedFacilities.length < 2) {
              selectedFacilities.push(facility);
            } else {
              selectedFacilities = [facility];
            }
          }
        } else {
          selectedFacilities = [];
          selectedFacility = facility;
        }
        renderFacilityInfo();
      });
    }
  });
}

function renderFacilityInfo() {
  const info = document.getElementById("info");
  const defaultMsg = document.getElementById("default-message");
  if (defaultMsg) defaultMsg.style.display = "none";

  if (compareMode) {
    if (selectedFacilities.length === 2) {
      const [a, b] = selectedFacilities;
      const nameA = encodeURIComponent(a.facilityName);
      const nameB = encodeURIComponent(b.facilityName);
      //const pathAB = `/vs_network/network_${nameA}_vs_${nameB}.html`;
      //const pathBA = `/vs_network/network_${nameB}_vs_${nameA}.html`;
      const pathAB = `/vs_network_v3/network_${nameA}_vs_${nameB}.html`;
      const pathBA = `/vs_network_v3/network_${nameB}_vs_${nameA}.html`;
      const pathA = `/network/cooccurrence_network_${nameA}_tfidf_v3.html`
      const pathB = `/network/cooccurrence_network_${nameB}_tfidf_v3.html`


      const iframe = document.createElement("iframe");
      iframe.style.width = "100%";
      iframe.style.height = "800px";
      iframe.src = pathAB;
      iframe.onerror = () => {
        iframe.src = pathBA;
      };

      //info.innerHTML = `
      //  <div style="display: flex; justify-content: space-between; align-items: center;">
      //    <h2 style="margin: 0;">比較: ${a.facilityName} vs ${b.facilityName}</h2>
      //   <button onclick="clearFacilityInfo()" style="font-size: 20px;">&times;</button>
      //  </div>
      //    <iframe src=${pathA} style="width: 50%; height: 500px;"></iframe>
      //    <iframe src=${pathB} style="width: 50%; height: 500px;"></iframe>
      //`;
      info.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <h2 style="margin: 0;">比較: ${a.facilityName} vs ${b.facilityName}</h2>
          <button onclick="clearFacilityInfo()" style="font-size: 20px;">&times;</button>
        </div>
        <div style="display: flex; gap: 1rem; margin-top: 1rem;">
          <iframe src="${pathA}" style="width: 50%; height: 500px; border: 1px solid #ccc;"></iframe>
          <iframe src="${pathB}" style="width: 50%; height: 500px; border: 1px solid #ccc;"></iframe>
        </div>
        <h3 class="mt-4">共通特徴の比較ネットワーク</h3>
        <iframe src="${pathAB}" style="width: 100%; height: 500px; border: 1px solid #ccc;"></iframe>
      `;

      //info.appendChild(iframe);
    } else if (selectedFacilities.length === 1) {
      info.innerHTML = `<h2>${selectedFacilities[0].facilityName}</h2>
      <p>もう1つ施設を選んで比較してください。</p>`;
    } else {
      info.innerHTML = `
        <p>
          選択した2つの保育所評価データ比較が表示されます。
        </p>
      `;
    }
  } else {
    if (!selectedFacility) {
      info.innerHTML = "";
      if (defaultMsg) defaultMsg.style.display = "block";
      return;
    }
    info.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center;">
      <h2>${selectedFacility.facilityName}</h2>
      <button onclick="clearFacilityInfo()" style="float:right; font-size: 20px;">&times;</button>
      </div>
      <iframe src="/network/cooccurrence_network_${selectedFacility.facilityName}_tfidf_v3.html" style="width: 100%; height: 800px;"></iframe>
    `;
  }
}

window.clearFacilityInfo = function () {
  selectedFacility = null;
  selectedFacilities = [];
  renderFacilityInfo();
}

function setupButtons() {
  const buttonContainer = document.getElementById("color-buttons");
  colors.forEach((color) => {
    const btn = document.createElement("button");
    btn.innerText = `Show ${color} pins`;
    btn.onclick = () => {
      selectedColor = color;
      map.eachLayer((layer) => {
        if (layer instanceof L.Marker) map.removeLayer(layer);
      });
      renderMarkers(allFacilities);
    };
    buttonContainer.appendChild(btn);
  });
  const showAllBtn = document.createElement("button");
  showAllBtn.innerText = "Show All pins";
  showAllBtn.onclick = () => {
    selectedColor = null;
    map.eachLayer((layer) => {
      if (layer instanceof L.Marker) map.removeLayer(layer);
    });
    renderMarkers(allFacilities);
  };
  buttonContainer.appendChild(showAllBtn);
}

function setupCompareToggle() {
  const toggle = document.getElementById("compare-mode-toggle");
  toggle.addEventListener("change", (e) => {
    compareMode = e.target.checked;

    if (compareMode) {
      if (selectedFacilities.length === 2) {
        renderFacilityInfo();
      } else {
        selectedFacility = null;
        renderFacilityInfo();
      }
    } else {
      if (selectedFacilities.length > 0) {
        selectedFacility = selectedFacilities[0];
      } else {
        selectedFacility = null;
      }
      selectedFacilities = [];
      renderFacilityInfo();
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  setupButtons();
  loadCSV();
  setupCompareToggle();
});
