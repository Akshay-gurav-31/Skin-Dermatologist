// Conversion maps
const lengthToMeter = { m:1, km:1000, cm:0.01, mm:0.001, in:0.0254, ft:0.3048, yd:0.9144, mi:1609.344 };
const areaToSqMeter = { m2:1, km2:1e6, cm2:0.0001, in2:0.00064516, ft2:0.09290304, acre:4046.8564224, ha:10000 };

const categories = {
  length: { map: lengthToMeter, units:[
    {k:"m",name:"Meter (m)"},{k:"km",name:"Kilometer (km)"},{k:"cm",name:"Centimeter (cm)"},
    {k:"mm",name:"Millimeter (mm)"},{k:"in",name:"Inch (in)"},{k:"ft",name:"Foot (ft)"},
    {k:"yd",name:"Yard (yd)"},{k:"mi",name:"Mile (mi)"}
  ]},
  area: { map: areaToSqMeter, units:[
    {k:"m2",name:"Square meter (m²)"},{k:"km2",name:"Square kilometer (km²)"},
    {k:"cm2",name:"Square centimeter (cm²)"},{k:"in2",name:"Square inch (in²)"},
    {k:"ft2",name:"Square foot (ft²)"},{k:"acre",name:"Acre"},{k:"ha",name:"Hectare (ha)"}
  ]}
};

// DOM elements
const categorySelect=document.getElementById('categorySelect');
const fromUnit=document.getElementById('fromUnit');
const toUnit=document.getElementById('toUnit');
const fromValue=document.getElementById('fromValue');
const resultEl=document.getElementById('result');
const errorEl=document.getElementById('error');
const convertBtn=document.getElementById('convertBtn');
const clearBtn=document.getElementById('clearBtn');
const swapBtn=document.getElementById('swapBtn');

function populateUnits(catKey){
  const cat=categories[catKey];
  fromUnit.innerHTML=""; toUnit.innerHTML="";
  cat.units.forEach(u=>{
    fromUnit.add(new Option(u.name,u.k));
    toUnit.add(new Option(u.name,u.k));
  });
  if(catKey==="length"){fromUnit.value="m";toUnit.value="km";} else {fromUnit.value="m2";toUnit.value="ft2";}
  resultEl.textContent="—"; errorEl.textContent="";
}

function formatNumber(x){
  if(!isFinite(x)) return String(x);
  if(Math.abs(x-Math.round(x))<1e-12) return String(Math.round(x));
  return Number(x).toPrecision(12).replace(/\.?0+$/,'');
}

function unitLabel(k){
  const map={m:"m",km:"km",cm:"cm",mm:"mm",in:"in",ft:"ft",yd:"yd",mi:"mi",
             m2:"m²",km2:"km²",cm2:"cm²",in2:"in²",ft2:"ft²",acre:"acre",ha:"ha"};
  return map[k]||k;
}

function convert(){
  errorEl.textContent="";
  const catKey=categorySelect.value;
  const valRaw=fromValue.value;
  if(valRaw===""){resultEl.textContent="—";return;}
  const val=Number(valRaw);
  if(!isFinite(val)){errorEl.textContent="Enter valid number.";return;}
  const from=fromUnit.value,to=toUnit.value,map=categories[catKey].map;
  const base=val*map[from]; const converted=base/map[to];
  resultEl.textContent=`${formatNumber(converted)} ${unitLabel(to)}`;
}

// Events
categorySelect.addEventListener('change',()=>{populateUnits(categorySelect.value);convert();});
[fromValue,fromUnit,toUnit].forEach(el=>el.addEventListener('input',convert));
convertBtn.addEventListener('click',e=>{e.preventDefault();convert();});
clearBtn.addEventListener('click',()=>{fromValue.value="";resultEl.textContent="—";errorEl.textContent="";});
swapBtn.addEventListener('click',()=>{[fromUnit.value,toUnit.value]=[toUnit.value,fromUnit.value];convert();});

// Init
populateUnits(categorySelect.value);
