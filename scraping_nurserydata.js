import data from "./nursery_school/nursery_school_url.json" assert { type: "json"};
let list_of_url = [];
data.nursery_url.forEach(part_url => {
	list_of_url.push(part_url.url)
})
console.log(list_of_url);

import { createRequire } from 'module';
const require = createRequire(import.meta.url)
import fs from 'fs';

const puppeteer = require('puppeteer');
let nursery_total_data = []
//const url_list = ['https://www.fukunavi.or.jp/fukunavi/controller?actionID=hyk&cmd=hyklstdtldigest&BEF_PRC=hyk&HYK_ID=2023003714&HYK_ID1=&HYK_ID2=&HYK_ID3=&HYK_ID4=&HYK_ID5=&JGY_CD1=&JGY_CD2=&JGY_CD3=&JGY_CD4=&JGY_CD5=&SCHSVCSBRCD=&SVCDBRCD=&PTN_CD=&SVCSBRCDALL=&SVCSBRCD=031&AREA1=&AREA2=&AREA3=&HYK_YR=&SCHHYK_YR=&NAME=&JGY_CD=1310500822&MODE=multi&DVS_CD=&SVCDBR_CD=22&SVCSBR_CD=&ROW=0&FROMDT=&SCH_ACTION=hyklst&KOHYO=&GEN=&HYKNEN=&LISTSVC=&ORDER=&HYK_DTL_CHK=&PRMCMT_CHK=&HYK_CHK=&JGY_CHK=&SVC_CHK=&DIG_MOVE_FLG=&MLT_SVCSBR_CD1=&MLT_SVCSBR_CD2=&MLT_SVCSBR_CD3=&MLT_SVCSBR_CD4=&MLT_SVCSBR_CD5=&MLT_SVCSBR_CD6=&MLT_SVCSBR_CD7=&MLT_SVCSBR_CD8=&COLOR_FLG=&COLOR_HYK_ID=&BEFORE_FLG=&MLT_DTL_SVCSBR_CD1=&MLT_DTL_SVCSBR_CD2=&MLT_DTL_SVCSBR_CD3=&MLT_DTL_SVCSBR_CD4=&MLT_DTL_SVCSBR_CD5=&MLT_DTL_SVCSBR_CD6=&MLT_DTL_SVCSBR_CD7=&MLT_DTL_SVCSBR_CD8=&HIKAKU_SVCSBRCD=&TELOPN001_NO1=&TELOPN001_NO2=&TELOPN001_NO3=&TELOPN002_NO1=&TELOPN002_NO2=&TELOPN002_NO3=&TELOPN003_NO1=&TELOPN003_NO2=&TELOPN003_NO3=&S_MODE=service&MLT_AREA=13105&H_NAME=&J_NAME=&SVCDBR_CD=22&STEP_SVCSBRCD=031&STEP_SVCSBRCD=032&STEP_SVCSBRCD=332&STEP_SVCSBRCD=437&STEP_SVCSBRCD=035&STEP_SVCSBRCD=037&STEP_SVCSBRCD=042&STEP_SVCSBRCD=036&STEP_SVCSBRCD=038']
const scrapeing_nurserydata = async () => {
  
  for (const url of list_of_url){
  	const browser = await puppeteer.launch();

  	const page = await browser.newPage();
  	await page.goto(url,{ waitUntil: 'domcontentloaded' });

  	const target = '.koumoku2';
  	const links = await page.$$eval(target, links => {
  	  return links.map(link => link.textContent);
  	});
  	const target2 = '.koumoku4';
  	const links2 = await page.$$eval(target2, links2 => {
  	  return links2.map(link2 => link2.textContent);
  	});

  	const target3 = '.yearbox';
  	const links3 = await page.$$eval(target3, links3 => {
  	  return links3.map(link3 => link3.textContent);
  	});
  	//console.log(links3[0].split('\n\t\t')[0])
  	//console.log(links3[0].split('\n\t\t')[1])

  	const target4 = 'li';
  	const links4 = await page.$$eval(target4, links4 => {
  	  	return links4.map(link4 => link4.textContent);
  	});

  	let nursery_data = {}
  	nursery_data['評価年度'] = links3[0].split('\n\t\t')[0].trim()
  	nursery_data['事業所種'] = links3[0].split('\n\t\t')[1].trim()
  	nursery_data['事業所名称'] = links[1].trim()
  	nursery_data['事業所の理念'] = links2[0].trim()
	nursery_data['サービス分析結果'] = links4.slice(7)
  	nursery_total_data.push(nursery_data)

  	console.log(nursery_total_data);
  	browser.close();
	
   }
	const json_string = JSON.stringify(nursery_total_data);
	console.log(json_string)
   	fs.writeFileSync('nursery_data.json',json_string)
}

scrapeing_nurserydata();
