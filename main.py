import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.statespace.sarimax import SARIMAX as tsa
import pandas as pd
import yfinance as yf
from datetime import datetime
from pmdarima import auto_arima


symbol_list=["AVOD.IS", "A1CAP.IS", "ACSEL.IS", "ADEL.IS", "ADESE.IS", "AFYON.IS", "AGHOL.IS", "AGESA.IS", "AHGAZ.IS", "AKSFA.IS", "AKFK.IS", "AKMEN.IS", "AKBNK.IS", "AKCNS.IS", "AKDFA.IS", "AKYHO.IS", "AKENR.IS", "AKFGY.IS", "AKFEN.IS", "ATEKS.IS", "AKSGY.IS", "AKMGY.IS", "AKSA.IS", "AKSEN.IS", "AKGRT.IS", "AKSUE.IS", "AKTVK.IS", "AFB.IS", "ALCAR.IS", "ALGYO.IS", "ALARK.IS", "ALBMD.IS", "ALBRK.IS", "ALCTL.IS", "ALFAS.IS", "ALJF.IS", "ALKIM.IS", "ALKA.IS", "ALNUS.IS", "ALNTF.IS", "AYCES.IS", "ALMAD.IS", "ANSGR.IS", "AEFES.IS", "ANHYT.IS", "ASUZU.IS", "ANGEN.IS", "ANELE.IS", "ARCLK.IS", "ARDYZ.IS", "ARENA.IS", "ARNFK.IS", "ARMDA.IS", "ARSAN.IS", "ARZUM.IS", "ASELS.IS", "ASTOR.IS", "ATAGY.IS", "ATA.IS", "AGYO.IS", "ATLFA.IS", "ATSYH.IS", "ATLAS.IS", "ATATP.IS", "AVGYO.IS", "AVTUR.IS", "AVHOL.IS", "AYDEM.IS", "AYEN.IS", "AYES.IS", "AYGAZ.IS", "AZTEK.IS", "AKFYE.IS", "BAGFS.IS", "BAKAB.IS", "BALAT.IS", "BNTAS.IS", "BANVT.IS", "BARMA.IS", "BASGZ.IS", "BASCM.IS", "BTCIM.IS", "BSOKE.IS", "BAYRK.IS", "BERA.IS", "BRKT.IS", "BRKSN.IS", "BJKAS.IS", "BEYAZ.IS", "BIENF.IS", "BLCYT.IS", "BLKOM.IS", "BIMAS.IS", "BIOEN.IS", "BRKVY.IS", "BRKO.IS", "BRLSM.IS", "BRMEN.IS", "BIZIM.IS", "BMSTL.IS", "BMSCH.IS", "BNPFK.IS", "BOBET.IS", "BRSAN.IS", "BRYAT.IS", "BFREN.IS", "BOSSA.IS", "BRISA.IS", "BURCE.IS", "BURVA.IS", "BUCIM.IS", "BVSAN.IS", "BIENY.IS", "BIGCH.IS", "CRFSA.IS", "CASA.IS", "CEOEM.IS", "CREAN.IS", "CCOLA.IS", "CONSE.IS", "COSMO.IS", "CRDFA.IS", "CAGFA.IS", "CLDNM.IS", "CLKEN.IS", "CANTE.IS", "CLEBI.IS", "CELHA.IS", "CEMAS.IS", "CEMTS.IS", "CMBTN.IS", "CMENT.IS", "CIMSA.IS", "CUSAN.IS", "CWENE.IS", "CVKMD.IS", "DYBNK.IS", "DAGI.IS", "DAGHL.IS", "DAPGM.IS", "DARDL.IS", "DGATE.IS", "DGRVK.IS", "DMSAS.IS", "DENGE.IS", "DENFA.IS", "DNFIN.IS", "DZGYO.IS", "DZY.IS", "DENIZ.IS", "DERIM.IS", "DERHL.IS", "DESA.IS", "DESPC.IS", "DTYGD.IS", "DEVA.IS", "DNISI.IS", "DIRIT.IS", "DITAS.IS", "DOCO.IS", "DOBUR.IS", "DDTCR.IS", "DOHOL.IS", "DTRND.IS", "DGNMO.IS", "ARASE.IS", "DOGUB.IS", "DGGYO.IS", "DOAS.IS", "DFKTR.IS", "DOKTA.IS", "DURDO.IS", "DNYVA.IS", "DYOBY.IS", "EDATAGY.IS", "ECZYT.IS", "EDIP.IS", "EGEEN.IS", "EGGUB.IS", "EGPRO.IS", "EGSER.IS", "EPLAS.IS", "ECILC.IS", "EKIZ.IS", "EKOFA.IS", "ELITE.IS", "EMKEL.IS", "EMNIS.IS", "EMIRV.IS", "EKTVK.IS", "EKGYO.IS", "EMVAR.IS", "ENJSA.IS", "ENKAI.IS", "ENSRI.IS", "ERBOS.IS", "ERCB.IS", "EREGL.IS", "ERGLI.IS", "KIMMR.IS", "ERSU.IS", "ESCAR.IS", "ESCOM.IS", "ESEN.IS", "ETILR.IS", "EUKYO.IS", "EUYO.IS", "ETYAT.IS", "EUHOL.IS", "TEZOL.IS", "EUREN.IS", "EYGYO.IS", "EUPWR.IS", "EKSUN.IS", "FADE.IS", "FSDAT.IS", "FMIZP.IS", "FENER.IS", "FIBAF.IS", "FBB.IS", "FLAP.IS", "FONET.IS", "FROTO.IS", "FORMT.IS", "FRIGO.IS", "GWIND.IS", "GSRAY.IS", "GAPIN.IS", "GARFA.IS", "GARFL.IS", "GRFIN.IS", "GRNYO.IS", "GEDIK.IS", "GEDZA.IS", "GLCVY.IS", "GENIL.IS", "GENTS.IS", "GEREL.IS", "GZNMI.IS", "GMTAS.IS", "GESAN.IS", "GLB.IS", "GLYHO.IS", "GGBVK.IS", "GSIPD.IS", "GOODY.IS", "GOLTS.IS", "GOZDE.IS", "GSDDE.IS", "GSDHO.IS", "GUBRF.IS", "GLRYH.IS", "GRSEL.IS", "GRTRK.IS", "GOKNR.IS", "SAHOL.IS", "HALKF.IS", "HLGYO.IS", "HLVKS.IS", "HALKI.IS", "HATEK.IS", "HDFFL.IS", "HDFGS.IS", "HEDEF.IS", "HEKTS.IS", "HKTM.IS", "HTTBT.IS", "HSB.IS", "HUBVC.IS", "HUNER.IS", "HUZFA.IS", "HURGZ.IS", "ICB.IS", "GIN.IS", "INVEO.IS", "INVES.IS", "ISKPL.IS", "IEYHO.IS", "IDEAS.IS", "IDGYO.IS", "IHEVA.IS", "IHLGM.IS", "IHGZT.IS", "IHAAS.IS", "IHLAS.IS", "IHYAY.IS", "IMASM.IS", "INALR.IS", "INDES.IS", "INFO.IS", "INTEM.IS", "IPEKE.IS", "ISDMR.IS", "ISFAK.IS", "ISFIN.IS", "ISGYO.IS", "ISGSY.IS", "ISMEN.IS", "ISYAT.IS", "ISBIR.IS", "ISSEN.IS", "ITTFH.IS", "IZINV.IS", "IZMDC.IS", "IZFAS.IS", "JANTS.IS", "KFEIN.IS", "KLKIM.IS", "KLVKS.IS", "KAPTESTAS001.IS", "KAPLM.IS", "KRDMA.IS", "KAREL.IS", "KARSN.IS", "KRTEK.IS", "KARYE.IS", "KARTN.IS", "KATVK.IS", "KATMR.IS", "KNTFA.IS", "KENT.IS", "KERVT.IS", "KRVGD.IS", "KERVN.IS", "KZBGY.IS", "KLGYO.IS", "KLRHO.IS", "KMPUR.IS", "KLMSN.IS", "KCAER.IS", "KFKTF.IS", "KOCFN.IS", "KCHOL.IS", "KLSYN.IS", "KNFRT.IS", "KONTR.IS", "KONYA.IS", "KONKA.IS", "KGYO.IS", "KORDS.IS", "KRPLS.IS", "KORTS.IS", "KOZAL.IS", "KOZAA.IS", "KRGYO.IS", "KRSTL.IS", "KRONT.IS", "KTKVK.IS", "KSTUR.IS", "KUVVA.IS", "KUYAS.IS", "KUTPO.IS", "KTSKR.IS", "KAYSE.IS", "KOPOL.IS", "LIDER.IS", "LIDFA.IS", "LINK.IS", "LOGO.IS", "LKMNH.IS", "LUKSK.IS", "MACKO.IS", "MAKIM.IS", "MAKTK.IS", "MANAS.IS", "MRBAS.IS", "MAGEN.IS", "MARKA.IS", "MAALT.IS", "MRSHL.IS", "MRGYO.IS", "MARTI.IS", "MTRKS.IS", "MAVI.IS", "MZHLD.IS", "MEDTR.IS", "MEGAP.IS", "MEKMD.IS", "MNDRS.IS", "MEPET.IS", "MERCN.IS", "MBFTR.IS", "MERIT.IS", "MERKO.IS", "METUR.IS", "METRO.IS", "MTRYO.IS", "MIATK.IS", "MGROS.IS", "MIPAZ.IS", "MSGYO.IS", "MPARK.IS", "MMCAS.IS", "MOBTL.IS", "MNDTR.IS", "EGEPO.IS", "NATEN.IS", "NTGAZ.IS", "NTHOL.IS", "NETAS.IS", "NIBAS.IS", "NUHCM.IS", "NUGYO.IS", "NRHOL.IS", "NRLIN.IS", "NURVK.IS", "NRBNK.IS", "OBASE.IS", "ODAS.IS", "ONCSM.IS", "OPET.IS", "ORCAY.IS", "ORFIN.IS", "ORGE.IS", "ORMA.IS", "OMD.IS", "OSTIM.IS", "OTKAR.IS", "OTOKC.IS", "OTTO.IS", "OYAKC.IS", "OYA.IS", "OYAYO.IS", "OYLUM.IS", "OZKGY.IS", "OZGYO.IS", "OZRDN.IS", "OZSUB.IS", "PALEN.IS", "PLGAZ.IS", "PAMEL.IS", "PNLSN.IS", "PAGYO.IS", "PAPIL.IS", "PRDGS.IS", "PRKME.IS", "PARSN.IS", "PBT.IS", "PSGYO.IS", "PCILT.IS", "PGSUS.IS", "PEKGY.IS", "PENGD.IS", "PENTA.IS", "PEGYO.IS", "PSDTC.IS", "PETKM.IS", "PKENT.IS", "PHC.IS", "PETUN.IS", "PINSU.IS", "PNSUT.IS", "PKART.IS", "PLTUR.IS", "POLHO.IS", "POLTK.IS", "PRZMA.IS", "QYHOL.IS", "QNBFF.IS", "QNBFL.IS", "QNBVK.IS", "FNY.IS", "FIN.IS", "QUAGR.IS", "RNPOL.IS", "RALYH.IS", "RAYSG.IS", "RYGYO.IS", "RYSAS.IS", "RHEAG.IS", "RODRG.IS", "ROYAL.IS", "RTALB.IS", "RUBNS.IS", "SAFKR.IS", "SANEL.IS", "SNICA.IS", "SANFM.IS", "SANKO.IS", "SAMAT.IS", "SARKY.IS", "SARTN.IS", "SASA.IS", "SAYAS.IS", "SDTTR.IS", "SEKUR.IS", "SELEC.IS", "SELGD.IS", "SELVA.IS", "SNKRN.IS", "SRVGY.IS", "SEYKM.IS", "SILVR.IS", "SNGYO.IS", "SMRTG.IS", "SMART.IS", "SODSN.IS", "SOKE.IS", "SKTAS.IS", "SONME.IS", "SNPAM.IS", "SUMAS.IS", "SUNTK.IS", "SUWEN.IS", "SZUKI.IS", "SMRFA.IS", "SMRVA.IS", "SEKFA.IS", "SEKFK.IS", "SEGYO.IS", "SKY.IS", "SEK.IS", "SOKM.IS", "DRPHN.IS", "TOKI.IS", "TAC.IS", "TAMFA.IS", "TNZTP.IS", "TATGD.IS", "TAVHL.IS", "TEBFA.IS", "TEBCE.IS", "TBY.IS", "TEKTU.IS", "TKFEN.IS", "TKNSA.IS", "TMPOL.IS", "TERA.IS", "TETMT.IS", "TFNVK.IS", "TGSAS.IS", "TOASO.IS", "TRGYO.IS", "TLMAN.IS", "TSPOR.IS", "TDGYO.IS", "ISKMD.IS", "TSGYO.IS", "TUCLK.IS", "TUKAS.IS", "TRCAS.IS", "TUREX.IS", "TRILC.IS", "FNCLL.IS", "TCELL.IS", "TMSN.IS", "TUPRS.IS", "THYAO.IS", "PRKAB.IS", "TTKOM.IS", "TTRAK.IS", "TBORG.IS", "TURGG.IS", "GARAN.IS", "HALKB.IS", "EXIMB.IS", "ISATR.IS", "KLN.IS", "TSK.IS", "TURSG.IS", "SISE.IS", "TVB.IS", "UFUK.IS", "ULAS.IS", "ULUFA.IS", "ULUSE.IS", "ULUUN.IS", "UMPAS.IS", "USAK.IS", "UZERB.IS", "ULKER.IS", "UNLUS.IS", "UNLU.IS", "VAKFA.IS", "VAKFN.IS", "VKGYO.IS", "VKFYO.IS", "VAKVK.IS", "VAKKO.IS", "VANGD.IS", "VBTYZ.IS", "VDFLO.IS", "VERUS.IS", "VERTU.IS", "VESBE.IS", "VESTL.IS", "VKING.IS", "VDFAS.IS", "YKFKT.IS", "YKR.IS", "YKB.IS", "YAPRK.IS", "YATAS.IS", "YAT.IS", "YATVK.IS", "YYLGD.IS", "YAYLA.IS", "YGGYO.IS", "YEOTK.IS", "YGYO.IS", "YYAPI.IS", "YESIL.IS", "YBTAS.IS", "YONGA.IS", "YKSLN.IS", "YUNSA.IS", "ZEDUR.IS", "ZRGYO.IS", "ZKBVK.IS", "ZOREN.IS"]

crsymbol_list=["BNB-USD", "XRP-USD", "SOL-USD", "BCH-USD", "LTC-USD", "TRB-USD", "DOGE-USD", "MKR-USD", "LINK-USD", "DOT-USD", "BNBUSD_PERP-USD", "BLZ-USD", "MATIC-USD", "APE-USD", "ADA-USD", "AVAX-USD", "1000SHIB-USD", "STORJ-USD", "RUNE-USD", "ATOM-USD", "APT-USD", "TRX-USD", "CRV-USD", "FIL-USD", "EOS-USD", "ETC-USD", "AAVE-USD", "XMRUSD_PERP-USD", "DOGEUSD_PERP-USD", "UNI-USD", "XLM-USD", "AXS-USD", "XRPUSD_PERP-USD", "LDO-USD", "NEAR-USD", "FTM-USD", "GALA-USD", "COMP-USD", "MTL-USD", "TOMO-USD", "GMT-USD", "MASK-USD", "LTCUSD_PERP-USD", "SAND-USD", "BAKE-USD", "XMR-USD", "STMX-USD", "LINA-USD", "WAVES-USD", "VET-USD", "SNX-USD", "SFP-USD", "ALGO-USD", "FILUSD_PERP-USD", "ADAUSD_PERP-USD", "MANA-USD", "SUSHI-USD", "HBAR-USD", "1INCH-USD", "DOTUSD_PERP-USD", "FLM-USD", "AAVEUSD_PERP-USD", "OGN-USD", "CHZ-USD", "1000XEC-USD", "ENJ-USD", "STG-USD", "SXP-USD", "GRT-USD", "YFI-USD", "KNC-USD", "ARPA-USD", "KAVA-USD", "EGLD-USD", "NEO-USD", "SPELL-USD", "THETA-USD", "ZEC-USD", "XTZ-USD", "QNT-USD", "EOSUSD_PERP-USD", "IMX-USD", "LINKUSD_PERP-USD", "PEOPLE-USD", "ETCUSD_PERP-USD", "OCEAN-USD", "DASH-USD", "ROSE-USD", "CHR-USD", "C98-USD", "SOLUSD_PERP-USD", "REEF-USD", "ONT-USD", "ONE-USD", "ZIL-USD", "LRC-USD", "BCHUSD_PERP-USD", "GAL-USD", "BEL-USD", "WOO-USD", "OMG-USD", "COTI-USD", "IOTA-USD", "CTSI-USD", "OP-USD", "AUDIO-USD", "LUNA2-USD", "ANKR-USD", "KSM-USD", "RSR-USD", "DUSK-USD", "JASMY-USD", "BAND-USD", "ATA-USD", "XEM-USD", "BAT-USD", "ZRX-USD", "RVN-USD", "UNIUSD_PERP-USD", "ALPHA-USD", "QTUM-USD", "HOT-USD", "NKN-USD", "BAL-USD", "DENT-USD", "SKL-USD", "REN-USD", "RLC-USD", "CELR-USD", "ICX-USD", "IOST-USD", "CVX-USD", "IOTX-USD", "CTK-USD", "INJ-USD", "ATOMUSD_PERP-USD", "DYDX-USD", "DGB-USD", "AVAXUSD_PERP-USD", "TLM-USD", "UNFI-USD", "XLMUSD_PERP-USD", "FOOTBALL-USD", "TRXUSD_PERP-USD", "ENSUSD_PERP-USD", "LPT-USD", "AXSUSD_PERP-USD", "APTUSD_PERP-USD", "KLAY-USD", "FLOW-USD", "API3-USD", "THETAUSD_PERP-USD", "CELO-USD", "AR-USD", "ANT-USD", "LIT-USD", "GTC-USD", "XRPBUSD-USD", "ENS-USD", "OPUSD_PERP-USD", "BNX-USD", "ALICE-USD", "ZEN-USD", "EGLDUSD_PERP-USD", "DAR-USD", "MATICUSD_PERP-USD", "NEARUSD_PERP-USD", "BNBBUSD-USD", "APEUSD_PERP-USD", "ICP-USD", "RUNEUSD_PERP-USD", "BLUEBIRD-USD", "XTZUSD_PERP-USD", "SANDUSD_PERP-USD", "FTMUSD_PERP-USD", "MANAUSD_PERP-USD", "KNCUSD_PERP-USD", "ALGOUSD_PERP-USD", "GMTUSD_PERP-USD", "ICXUSD_PERP-USD", "CHZUSD_PERP-USD", "ROSEUSD_PERP-USD", "JOE-USD", "ARB-USD", "1000PEPE-USD", "PERP-USD", "SUI-USD", "STX-USD", "CFX-USD", "HOOK-USD", "RDNT-USD", "LEVER-USD", "HIGH-USD", "RNDR-USD", "FET-USD", "SSV-USD", "ID-USD", "GMX-USD", "TRU-USD", "AGIX-USD", "MAGIC-USD", "EDU-USD", "XVS-USD", "ASTR-USD", "BLUR-USD", "ACH-USD", "T-USD", "IDEX-USD", "AMB-USD", "FXS-USD", "PHB-USD", "MINA-USD", "LQTY-USD", "HFT-USD", "1000FLOKI-USD", "CKB-USD", "USDC-USD"]
end=datetime.now().strftime("%Y-%m-%d")
p,d,q=5,2,3
P,D,Q,s=2,0,2,7
def test():

    data = yf.download("BNB-USD", start="2010-01-01", end=end, progress=False, interval="1d")
    # time zone localize
    data.index = data.index.tz_localize(None)
    data.reset_index(inplace=True, names=['Date', 'Close'])
    data = data.fillna(0)

    data = data[['Date', 'Close']]
    model = auto_arima(data['Close'], seasonal=True, m=12, stepwise=True, suppress_warnings=True, trace=False)
    print(model.summary())


def main():
    result_df = pd.DataFrame(columns=['Symbol', 'Last Price', 'Forecast Price', 'Price Difference'])
    for symbol in symbol_list:

        try:
            data = yf.download(symbol, start="2018-01-01", end=end, progress=False, interval="1d")
            # time zone localize
            data.index = data.index.tz_localize(None)
            data.reset_index(inplace=True, names=['Date', 'Close'])
            data = data.fillna(0)

            data = data[['Date', 'Close']]
            model = tsa(data['Close'], order=(p, d, q),exogenous_variables=True, seasonal_order=(P, D, Q, s), enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=30)
            last_close_price = data['Close'].iloc[-1]
            if forecast.max().max() > last_close_price:
                forecast_pricex = forecast.max().max()
            else:
                forecast_pricex = forecast.iloc[0]
            forecast_price = forecast_pricex

            # Farkı hesaplayın
            price_difference = forecast_price - last_close_price
            price_difference_percent = (price_difference / last_close_price) * 100

            # Fiyat farkını pozitifse yeşil, negatifse kırmızı olarak biçimlendirin
            if price_difference < 0:
                price_color = "\x1b[31m"
            else:
                price_color = "\x1b[32m"

            # Biçimlendirilmiş metni oluşturun
            formatted_text = f"{symbol} last price: {last_close_price:.2f} --> forecast price: {price_color}{forecast_price:.2f}\x1b[0m, price difference: {price_color}{price_difference:.2f}\x1b[0m, price difference (%): {price_color}{price_difference_percent:.2f}%\x1b[0m"

            # Biçimlendirilmiş metni yazdırın
            print(formatted_text)

            # Veriyi sonuç DataFrame'ine ekleyin
            result_df = result_df._append({'Symbol': symbol,
                                           'Last Price': last_close_price,
                                           'Forecast Price': forecast_price,
                                           'Price Difference (%)': price_difference_percent}, ignore_index=True)
        except Exception as e:
            print(f"Hata ")

    top_5_hisse = result_df.sort_values(by='Price Difference (%)', ascending=False).head(5)
    print("\nEn çok yükselmesi beklenen 5 hisse:")
    print(top_5_hisse[['Symbol', 'Price Difference (%)']])

    result_df.to_excel('hisse_senedi_tahminleri.xlsx', index=False)
print("Başladı..")
main()
print("Sonlandı")


