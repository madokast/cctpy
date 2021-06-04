"""
CCT 建模优化代码
CosyMap 测试

作者：赵润晓
日期：2021年6月3日
"""

# 因为要使用父目录的 cctpy 所以加入
from os import error, path
import sys
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))
from cctpy import *
from opera_utils import *
from cosy_utils import *

SMALL_GAP_20201228_38 = CosyMap(map="""  0.3343354      2.049994     0.0000000E+00 0.0000000E+00-0.2089153E-01 100000
 -0.4429797     0.2748559     0.0000000E+00 0.0000000E+00-0.2670926E-01 010000
  0.0000000E+00 0.0000000E+00 0.9849251     0.1462247E-01 0.0000000E+00 001000
  0.0000000E+00 0.0000000E+00 0.1084427      1.016916     0.0000000E+00 000100
  0.0000000E+00 0.0000000E+00 0.0000000E+00 0.0000000E+00  1.000000     000010
 -0.1818438E-01-0.4901168E-01 0.0000000E+00 0.0000000E+00  2.169257     000001
   2.032566     0.4407782     0.0000000E+00 0.0000000E+00 -2.996100     200000
   1.329922      3.644530     0.0000000E+00 0.0000000E+00 -1.731224     110000
  0.2827519     -1.227376     0.0000000E+00 0.0000000E+00-0.6005224     020000
  0.0000000E+00 0.0000000E+00  30.23159      19.39502     0.0000000E+00 101000
  0.0000000E+00 0.0000000E+00  4.777806     -1.783247     0.0000000E+00 011000
  -3.827844      4.436364     0.0000000E+00 0.0000000E+00 -4.304213     002000
  0.0000000E+00 0.0000000E+00 -12.94025     -29.27018     0.0000000E+00 100100
  0.0000000E+00 0.0000000E+00  9.493409     -4.988388     0.0000000E+00 010100
   14.37588      2.484786     0.0000000E+00 0.0000000E+00 0.1297422     001100
  -3.314025     -1.849413     0.0000000E+00 0.0000000E+00-0.4237248     100001
  -1.192156     -2.134924     0.0000000E+00 0.0000000E+00-0.1523609     010001
  0.0000000E+00 0.0000000E+00 0.3536598      8.200657     0.0000000E+00 001001
 -0.5073111      11.82201     0.0000000E+00 0.0000000E+00 -2.113465     000200
  0.0000000E+00 0.0000000E+00 -3.633323     0.4838250     0.0000000E+00 000101
 -0.8751596E-01-0.9554060E-01 0.0000000E+00 0.0000000E+00 -2.162550     000002
   175.7860     -8.219388     0.0000000E+00 0.0000000E+00  10.40401     300000
   81.59080      58.60351     0.0000000E+00 0.0000000E+00  3.088839     210000
   22.49616      8.882810     0.0000000E+00 0.0000000E+00  9.090974     120000
  -1.038466     -11.14553     0.0000000E+00 0.0000000E+00  1.491763     030000
  0.0000000E+00 0.0000000E+00  154.0333     -430.4041     0.0000000E+00 201000
  0.0000000E+00 0.0000000E+00  192.3000      6.636235     0.0000000E+00 111000
  0.0000000E+00 0.0000000E+00 -44.23757     -45.89913     0.0000000E+00 021000
   228.7845      64.09729     0.0000000E+00 0.0000000E+00  42.55898     102000
  -11.90121      92.42823     0.0000000E+00 0.0000000E+00 -15.81575     012000
  0.0000000E+00 0.0000000E+00  54.23575     -57.50478     0.0000000E+00 003000
  0.0000000E+00 0.0000000E+00  1269.857      456.0378     0.0000000E+00 200100
  0.0000000E+00 0.0000000E+00  1046.510      323.1961     0.0000000E+00 110100
  0.0000000E+00 0.0000000E+00  176.1156      50.24559     0.0000000E+00 020100
  -107.5639      27.96095     0.0000000E+00 0.0000000E+00  64.84654     101100
  -100.5988     -50.96575     0.0000000E+00 0.0000000E+00 -28.14756     011100
  0.0000000E+00 0.0000000E+00  225.7635     -34.81505     0.0000000E+00 002100
   3.409729      19.94880     0.0000000E+00 0.0000000E+00  37.09991     200001
   3.903160      17.04740     0.0000000E+00 0.0000000E+00  17.63726     110001
   4.239141     -2.887154     0.0000000E+00 0.0000000E+00  5.014751     020001
  0.0000000E+00 0.0000000E+00 -80.75317     -356.1959     0.0000000E+00 101001
  0.0000000E+00 0.0000000E+00  23.02753     -39.89510     0.0000000E+00 011001
   129.9661     -21.35954     0.0000000E+00 0.0000000E+00  37.53898     002001
   692.2915      385.2558     0.0000000E+00 0.0000000E+00 -98.14611     100200
   371.4516      173.8010     0.0000000E+00 0.0000000E+00  5.164416     010200
  0.0000000E+00 0.0000000E+00  146.2408     -147.4376     0.0000000E+00 001200
  0.0000000E+00 0.0000000E+00 -293.1550     -143.8245     0.0000000E+00 100101
  0.0000000E+00 0.0000000E+00  3.182549      56.94547     0.0000000E+00 010101
   43.71202      88.69607     0.0000000E+00 0.0000000E+00  49.11147     001101
   18.60151      3.435679     0.0000000E+00 0.0000000E+00  2.536551     100002
   6.596218      6.033266     0.0000000E+00 0.0000000E+00 0.2562855     010002
  0.0000000E+00 0.0000000E+00  4.527231     -32.71563     0.0000000E+00 001002
  0.0000000E+00 0.0000000E+00  76.18423     -170.4989     0.0000000E+00 000300
  -113.1147      17.33349     0.0000000E+00 0.0000000E+00 -14.24691     000201
  0.0000000E+00 0.0000000E+00 -10.26147     -38.85410     0.0000000E+00 000102
  0.2833602    -0.2534549     0.0000000E+00 0.0000000E+00  2.312088     000003
   454.4044     -872.4770     0.0000000E+00 0.0000000E+00 -994.1188     400000
   373.9316     -921.6380     0.0000000E+00 0.0000000E+00 -1000.517     310000
  -499.0266     -150.9324     0.0000000E+00 0.0000000E+00 -382.8516     220000
  -220.8997     -279.4448     0.0000000E+00 0.0000000E+00 -18.65452     130000
  -40.23437     -38.92595     0.0000000E+00 0.0000000E+00 -8.104235     040000
  0.0000000E+00 0.0000000E+00  18982.20      13674.37     0.0000000E+00 301000
  0.0000000E+00 0.0000000E+00  5652.588      5225.448     0.0000000E+00 211000
  0.0000000E+00 0.0000000E+00 -2073.600      2231.750     0.0000000E+00 121000
  0.0000000E+00 0.0000000E+00 -1036.745     -91.63835     0.0000000E+00 031000
  -4809.171     -1125.992     0.0000000E+00 0.0000000E+00 -946.8561     202000
  -1318.098     -687.9908     0.0000000E+00 0.0000000E+00 -617.7914     112000
  -513.1125      641.1409     0.0000000E+00 0.0000000E+00 -246.0642     022000
  0.0000000E+00 0.0000000E+00  2519.820      3446.628     0.0000000E+00 103000
  0.0000000E+00 0.0000000E+00  357.4900      571.6583     0.0000000E+00 013000
  -1060.717      381.6631     0.0000000E+00 0.0000000E+00 -283.8613     004000
  0.0000000E+00 0.0000000E+00 -7140.705      3041.306     0.0000000E+00 300100
  0.0000000E+00 0.0000000E+00  11055.67      3338.632     0.0000000E+00 210100
  0.0000000E+00 0.0000000E+00  3980.229      1235.296     0.0000000E+00 120100
  0.0000000E+00 0.0000000E+00 -499.0515     -176.1431     0.0000000E+00 030100
   12074.83      1247.721     0.0000000E+00 0.0000000E+00 -1691.874     201100
  -353.3677     -2048.310     0.0000000E+00 0.0000000E+00 -107.4114     111100
  -2591.175     -805.2561     0.0000000E+00 0.0000000E+00 -481.1725     021100
  0.0000000E+00 0.0000000E+00  14071.19      6123.696     0.0000000E+00 102100
  0.0000000E+00 0.0000000E+00  3218.449     -493.5128     0.0000000E+00 012100
  -1867.057     -693.8021     0.0000000E+00 0.0000000E+00 -1057.968     003100
  -2589.761     -601.3165     0.0000000E+00 0.0000000E+00 -213.8633     300001
  -2089.317     -561.1465     0.0000000E+00 0.0000000E+00 -64.93837     210001
  -494.0103     -187.2277     0.0000000E+00 0.0000000E+00 -81.69040     120001
  -48.16541     -87.04480     0.0000000E+00 0.0000000E+00 -6.715259     030001
  0.0000000E+00 0.0000000E+00  11712.62      10179.10     0.0000000E+00 201001
  0.0000000E+00 0.0000000E+00  6534.847      3536.730     0.0000000E+00 111001
  0.0000000E+00 0.0000000E+00  953.7302      683.1657     0.0000000E+00 021001
  -3991.029      194.8343     0.0000000E+00 0.0000000E+00 -267.0478     102001
  -963.7898     -1249.931     0.0000000E+00 0.0000000E+00 -79.35998     012001
  0.0000000E+00 0.0000000E+00  304.3814      506.1289     0.0000000E+00 003001
  -9949.822     -1657.050     0.0000000E+00 0.0000000E+00 -4630.090     200200
   503.0146      1368.352     0.0000000E+00 0.0000000E+00 -1247.587     110200
  -500.5654     -714.3852     0.0000000E+00 0.0000000E+00 -64.85167     020200
  0.0000000E+00 0.0000000E+00  5387.677      4470.796     0.0000000E+00 101200
  0.0000000E+00 0.0000000E+00  1586.424      2792.176     0.0000000E+00 011200
   3009.142      2117.183     0.0000000E+00 0.0000000E+00 -624.3821     002200
  0.0000000E+00 0.0000000E+00 -12507.10      1685.177     0.0000000E+00 200101
  0.0000000E+00 0.0000000E+00 -6657.836     -1068.663     0.0000000E+00 110101
  0.0000000E+00 0.0000000E+00 -523.7806      172.2017     0.0000000E+00 020101
   8769.582      2322.297     0.0000000E+00 0.0000000E+00 -1626.366     101101
   5797.438      1063.038     0.0000000E+00 0.0000000E+00  187.3438     011101
  0.0000000E+00 0.0000000E+00 -695.1399      1466.687     0.0000000E+00 002101
  -88.13152     -26.30176     0.0000000E+00 0.0000000E+00 -137.9364     200002
  -19.22725     -99.69130     0.0000000E+00 0.0000000E+00 -98.09034     110002
  -14.06663      12.84339     0.0000000E+00 0.0000000E+00 -27.02077     020002
  0.0000000E+00 0.0000000E+00 -643.8479      1695.893     0.0000000E+00 101002
  0.0000000E+00 0.0000000E+00 -108.2759      599.7408     0.0000000E+00 011002
  -520.1975      518.0473     0.0000000E+00 0.0000000E+00 -58.65282     002002
  0.0000000E+00 0.0000000E+00  20690.50      5711.653     0.0000000E+00 100300
  0.0000000E+00 0.0000000E+00  3801.255     -168.7067     0.0000000E+00 010300
  -1984.735     -3337.583     0.0000000E+00 0.0000000E+00 -672.9862     001300
  -5279.402     -1397.625     0.0000000E+00 0.0000000E+00 -758.5772     100201
  -307.6425      488.2117     0.0000000E+00 0.0000000E+00  360.6806     010201
  0.0000000E+00 0.0000000E+00 -1550.256     -1837.934     0.0000000E+00 001201
  0.0000000E+00 0.0000000E+00 -409.4044      1364.740     0.0000000E+00 100102
  0.0000000E+00 0.0000000E+00  48.77747      235.1094     0.0000000E+00 010102
  -1689.225     -296.5347     0.0000000E+00 0.0000000E+00 -326.7006     001102
  -38.36059      7.395293     0.0000000E+00 0.0000000E+00 -5.398468     100003
  -27.60781     -17.69473     0.0000000E+00 0.0000000E+00-0.2048923     010003
  0.0000000E+00 0.0000000E+00 -24.67562      52.10352     0.0000000E+00 001003
   6354.302      1208.293     0.0000000E+00 0.0000000E+00  7.108870     000400
  0.0000000E+00 0.0000000E+00 -155.1170      577.5242     0.0000000E+00 000301
  -339.1400     -410.0611     0.0000000E+00 0.0000000E+00 -257.5990     000202
  0.0000000E+00 0.0000000E+00 -96.95883      76.74887     0.0000000E+00 000103
  0.3084108E-01  1.867732     0.0000000E+00 0.0000000E+00 -2.827693     000004
   54186.67      20097.32     0.0000000E+00 0.0000000E+00  5903.382     500000
   79939.88      14323.37     0.0000000E+00 0.0000000E+00  1038.208     410000
   32332.15      9681.964     0.0000000E+00 0.0000000E+00  1142.906     320000
   3174.225      3488.706     0.0000000E+00 0.0000000E+00  1909.729     230000
   250.2696     -515.1542     0.0000000E+00 0.0000000E+00  661.9620     140000
   57.79449      237.2344     0.0000000E+00 0.0000000E+00 -36.20222     050000
  0.0000000E+00 0.0000000E+00 -333715.4     -178877.1     0.0000000E+00 401000
  0.0000000E+00 0.0000000E+00 -134646.4     -120825.5     0.0000000E+00 311000
  0.0000000E+00 0.0000000E+00 -182691.7     -106861.5     0.0000000E+00 221000
  0.0000000E+00 0.0000000E+00 -72095.30     -24842.80     0.0000000E+00 131000
  0.0000000E+00 0.0000000E+00 -7311.614     -4205.817     0.0000000E+00 041000
   162760.8      7429.709     0.0000000E+00 0.0000000E+00 -3493.792     302000
   42478.21      5076.150     0.0000000E+00 0.0000000E+00  946.5236     212000
   30895.69      13942.63     0.0000000E+00 0.0000000E+00  876.0456     122000
   9355.165      11046.58     0.0000000E+00 0.0000000E+00  411.3578     032000
  0.0000000E+00 0.0000000E+00 -28419.96     -68820.28     0.0000000E+00 203000
  0.0000000E+00 0.0000000E+00 -57813.80     -34918.36     0.0000000E+00 113000
  0.0000000E+00 0.0000000E+00 -17738.27     -2190.242     0.0000000E+00 023000
   27255.27     -12091.75     0.0000000E+00 0.0000000E+00  1662.957     104000
   8615.635      12652.91     0.0000000E+00 0.0000000E+00  1176.016     014000
  0.0000000E+00 0.0000000E+00 -1287.218     -361.6815     0.0000000E+00 005000
  0.0000000E+00 0.0000000E+00 -228412.2     -154274.8     0.0000000E+00 400100
  0.0000000E+00 0.0000000E+00  155847.6     -19633.57     0.0000000E+00 310100
  0.0000000E+00 0.0000000E+00  152596.5     -8139.166     0.0000000E+00 220100
  0.0000000E+00 0.0000000E+00 -19717.29     -7683.967     0.0000000E+00 130100
  0.0000000E+00 0.0000000E+00 -11451.12     -3410.598     0.0000000E+00 040100
  -448069.1     -83567.40     0.0000000E+00 0.0000000E+00 -53951.89     301100
  -145251.3     -9820.441     0.0000000E+00 0.0000000E+00  5144.254     211100
  -115667.0     -28778.55     0.0000000E+00 0.0000000E+00  14451.43     121100
  -27931.74     -693.1563     0.0000000E+00 0.0000000E+00 -513.2434     031100
  0.0000000E+00 0.0000000E+00  129151.6     -62944.50     0.0000000E+00 202100
  0.0000000E+00 0.0000000E+00  251769.2      72071.55     0.0000000E+00 112100
  0.0000000E+00 0.0000000E+00  36882.05     -6345.878     0.0000000E+00 022100
   23072.64      8691.627     0.0000000E+00 0.0000000E+00  23883.81     103100
  -89683.58     -15870.17     0.0000000E+00 0.0000000E+00 -4482.748     013100
  0.0000000E+00 0.0000000E+00  26573.18     -16177.05     0.0000000E+00 004100
   7253.585      3854.984     0.0000000E+00 0.0000000E+00  5464.504     400001
  -4062.709      3846.298     0.0000000E+00 0.0000000E+00  9033.070     310001
   1668.176      3574.416     0.0000000E+00 0.0000000E+00  4585.996     220001
   2899.188      1297.777     0.0000000E+00 0.0000000E+00  613.4505     130001
   315.1976     -268.6204     0.0000000E+00 0.0000000E+00  113.6900     040001
  0.0000000E+00 0.0000000E+00 -333315.1     -173182.5     0.0000000E+00 301001
  0.0000000E+00 0.0000000E+00 -50265.39     -77013.66     0.0000000E+00 211001
  0.0000000E+00 0.0000000E+00  23246.04     -29432.90     0.0000000E+00 121001
  0.0000000E+00 0.0000000E+00  3358.347     -3729.988     0.0000000E+00 031001
   163245.6      11749.81     0.0000000E+00 0.0000000E+00  1983.360     202001
   10595.14      4580.844     0.0000000E+00 0.0000000E+00  9322.112     112001
  -13220.70     -13553.23     0.0000000E+00 0.0000000E+00  1094.884     022001
  0.0000000E+00 0.0000000E+00  64976.16     -15957.54     0.0000000E+00 103001
  0.0000000E+00 0.0000000E+00  18398.92     -18051.26     0.0000000E+00 013001
   5269.574     -14746.54     0.0000000E+00 0.0000000E+00  111.2067     004001
   25926.81     -18288.09     0.0000000E+00 0.0000000E+00  59229.02     300200
   104226.3      7960.942     0.0000000E+00 0.0000000E+00  12672.85     210200
   9287.000     -19680.16     0.0000000E+00 0.0000000E+00  8771.831     120200
  -18889.62     -11133.87     0.0000000E+00 0.0000000E+00 -594.6155     030200
  0.0000000E+00 0.0000000E+00 -167405.9     -101226.6     0.0000000E+00 201200
  0.0000000E+00 0.0000000E+00 -213277.1     -35599.71     0.0000000E+00 111200
  0.0000000E+00 0.0000000E+00 -33828.50      177.6669     0.0000000E+00 021200
   22536.02     -8562.077     0.0000000E+00 0.0000000E+00 -1503.418     102200
   49701.01      35072.04     0.0000000E+00 0.0000000E+00 -7946.673     012200
  0.0000000E+00 0.0000000E+00  39453.84      29181.15     0.0000000E+00 003200
  0.0000000E+00 0.0000000E+00 -214340.2     -153347.8     0.0000000E+00 300101
  0.0000000E+00 0.0000000E+00 -55669.62     -65528.78     0.0000000E+00 210101
  0.0000000E+00 0.0000000E+00  20289.71     -26678.63     0.0000000E+00 120101
  0.0000000E+00 0.0000000E+00  6055.176     -1403.662     0.0000000E+00 030101
  -363647.1     -91408.29     0.0000000E+00 0.0000000E+00 -44380.59     201101
  -26142.64      26588.32     0.0000000E+00 0.0000000E+00 -14374.45     111101
   16927.70     -2325.596     0.0000000E+00 0.0000000E+00  1109.940     021101
  0.0000000E+00 0.0000000E+00 -142786.7     -111814.1     0.0000000E+00 102101
  0.0000000E+00 0.0000000E+00 -26331.53      8912.558     0.0000000E+00 012101
   114990.4      28453.42     0.0000000E+00 0.0000000E+00  13987.20     003101
   7482.990      1265.002     0.0000000E+00 0.0000000E+00  410.9429     300002
   11940.79      3114.868     0.0000000E+00 0.0000000E+00  716.9745     210002
   4289.988      997.7282     0.0000000E+00 0.0000000E+00  278.2728     120002
   503.4447      504.8518     0.0000000E+00 0.0000000E+00  12.02745     030002
  0.0000000E+00 0.0000000E+00 -136312.3     -69487.91     0.0000000E+00 201002
  0.0000000E+00 0.0000000E+00 -81984.43     -39699.59     0.0000000E+00 111002
  0.0000000E+00 0.0000000E+00 -11143.68     -5222.095     0.0000000E+00 021002
   70575.43      723.3433     0.0000000E+00 0.0000000E+00 -2573.875     102002
   30446.57      12460.01     0.0000000E+00 0.0000000E+00  854.0780     012002
  0.0000000E+00 0.0000000E+00 -4002.906      1835.978     0.0000000E+00 003002
  0.0000000E+00 0.0000000E+00  623188.4      179990.9     0.0000000E+00 200300
  0.0000000E+00 0.0000000E+00  404309.9      98592.96     0.0000000E+00 110300
  0.0000000E+00 0.0000000E+00  45186.85      21563.28     0.0000000E+00 020300
  -251395.2     -140091.8     0.0000000E+00 0.0000000E+00  16894.15     101300
  -182169.1     -43836.40     0.0000000E+00 0.0000000E+00 -4362.733     011300
  0.0000000E+00 0.0000000E+00  71139.84      8694.928     0.0000000E+00 002300
  -122935.7     -16556.05     0.0000000E+00 0.0000000E+00  44591.29     200201
   24892.91      22686.38     0.0000000E+00 0.0000000E+00  26558.61     110201
   28186.49      6917.523     0.0000000E+00 0.0000000E+00  6522.482     020201
  0.0000000E+00 0.0000000E+00  195984.5     -1203.986     0.0000000E+00 101201
  0.0000000E+00 0.0000000E+00  33888.16     -9528.582     0.0000000E+00 011201
  -35423.89     -36751.32     0.0000000E+00 0.0000000E+00  10862.55     002201
  0.0000000E+00 0.0000000E+00 -31510.73     -46279.59     0.0000000E+00 200102
  0.0000000E+00 0.0000000E+00  6252.868     -11060.36     0.0000000E+00 110102
  0.0000000E+00 0.0000000E+00  5167.761     -1208.754     0.0000000E+00 020102
  -99993.63     -40013.65     0.0000000E+00 0.0000000E+00  610.8601     101102
  -32496.54      2053.122     0.0000000E+00 0.0000000E+00  3125.685     011102
  0.0000000E+00 0.0000000E+00 -8256.883     -23683.39     0.0000000E+00 002102
  -15.67485     -199.7194     0.0000000E+00 0.0000000E+00  299.7041     200003
   203.7357      330.1093     0.0000000E+00 0.0000000E+00  390.7349     110003
  -26.57668     -56.63140     0.0000000E+00 0.0000000E+00  114.0550     020003
  0.0000000E+00 0.0000000E+00  965.8979     -5054.371     0.0000000E+00 101003
  0.0000000E+00 0.0000000E+00  932.1753     -2831.836     0.0000000E+00 011003
  -1150.185     -4198.382     0.0000000E+00 0.0000000E+00 -151.5434     002003
   44339.49      41794.49     0.0000000E+00 0.0000000E+00 -26684.24     100400
  -19420.53     -11190.58     0.0000000E+00 0.0000000E+00 -15165.69     010400
  0.0000000E+00 0.0000000E+00  40527.33      70217.71     0.0000000E+00 001400
  0.0000000E+00 0.0000000E+00 -7165.691      11401.51     0.0000000E+00 100301
  0.0000000E+00 0.0000000E+00 -26675.12     -27294.69     0.0000000E+00 010301
   127975.4      13274.14     0.0000000E+00 0.0000000E+00  3973.479     001301
  -35676.16     -6973.284     0.0000000E+00 0.0000000E+00  5366.151     100202
  -15179.13     -1400.182     0.0000000E+00 0.0000000E+00  1009.579     010202
  0.0000000E+00 0.0000000E+00  877.5239     -254.0867     0.0000000E+00 001202
  0.0000000E+00 0.0000000E+00  1262.824     -6132.455     0.0000000E+00 100103
  0.0000000E+00 0.0000000E+00  608.9697     -1965.639     0.0000000E+00 010103
   4368.899     -3781.586     0.0000000E+00 0.0000000E+00 -864.6287     001103
   38.49238     -76.84820     0.0000000E+00 0.0000000E+00 -3.893521     100004
   86.68569      39.79934     0.0000000E+00 0.0000000E+00 -4.413937     010004
  0.0000000E+00 0.0000000E+00 -304.8015     -70.28295     0.0000000E+00 001004
  0.0000000E+00 0.0000000E+00  43947.39      11109.07     0.0000000E+00 000500
   26368.42      23405.60     0.0000000E+00 0.0000000E+00  6483.963     000401
  0.0000000E+00 0.0000000E+00  6662.736      11225.63     0.0000000E+00 000302
   3752.908      909.0326     0.0000000E+00 0.0000000E+00  440.5101     000203
  0.0000000E+00 0.0000000E+00  29.21580     -188.1003     0.0000000E+00 000104
  -5.198439     -7.857675     0.0000000E+00 0.0000000E+00  3.857676     000005""")


# 相椭圆
number = 64
plane = PhaseSpaceParticle.XXP_PLANE
ps_5 = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_plane(
    plane,
    xMax=3.5*MM,
    xpMax=7.5*MM,
    delta=0.05,
    number=number
)

ps_0 = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_plane(
    plane,
    xMax=3.5*MM,
    xpMax=7.5*MM,
    delta=0.00,
    number=number
)

ps_m5 = PhaseSpaceParticle.phase_space_particles_along_positive_ellipse_in_plane(
    plane,
    xMax=3.5*MM,
    xpMax=7.5*MM,
    delta=-0.05,
    number=number
)

pss = [ps_m5, ps_0, ps_5]

# 转为能量分散
pps = [PhaseSpaceParticle.convert_delta_from_momentum_dispersion_to_energy_dispersion_for_list(
    ps, centerKineticEnergy_MeV=250) for ps in pss]

# cosy map apply
pps_end = [SMALL_GAP_20201228_38.apply_phase_space_particles(
    ps, order=5) for ps in pps]

# 转回动能分散
pps_end = [PhaseSpaceParticle.convert_delta_from_energy_dispersion_to_momentum_dispersion_for_list(
    ps, centerKineticEnergy_MeV=250) for ps in pps_end]

Plot2.plot(PhaseSpaceParticle.phase_space_particles_project_to_plane(
    pps_end[0], plane, convert_to_mm=True), describe='r.')
Plot2.plot(PhaseSpaceParticle.phase_space_particles_project_to_plane(
    pps_end[1], plane, convert_to_mm=True), describe='k.')
Plot2.plot(PhaseSpaceParticle.phase_space_particles_project_to_plane(
    pps_end[2], plane, convert_to_mm=True), describe='b.')

Plot2.info(x_label='mm', y_label='mr', title="x-plane", font_size=18)

Plot2.legend('dp-5', 'dp0', 'dp5')

Plot2.equal()

# 展示图片
Plot2.show()
