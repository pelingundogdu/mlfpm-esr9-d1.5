all: t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11 t12 t13 t14 t15 t16 t17 t18 t19 t20 t21 t22 t23 t24 t25 t26 t27 t28 t29 t30 t31 t32 t33 t34 t35 t36 t37 t38 t39 t40 t41 t42 t43 t44 t45 t46 t47 t48 t49 t50 t51 t52 t53 t54 t55 t56 t57 t58 t59 t60 t61 t62 t63 t64 t65 t66 t67 t68 t69 t70 t71 t72 t73 t74 t75 t76 t77 t78 t79 t80 t81 t82 t83 t84 t85 t86 t87 t88 t89 t90 t91 t92 t93 t94 t95 t96 t97 t98 t99 t100 t101 t102 t103 t104 t105 t106 t107 t108 t109 t110 t111 t112 t113 t114 t115 t116 t117 t118 t119 t120 t121 t122 t123 t124 t125 t126 t127 t128 t129 t130 t131 t132 t133 t134 t135 t136 t137 t138 t139 t140 t141 t142 t143 t144 t145 t146 t147 t148 t149 t150 t151 t152 t153 t154 t155 t156 t157 t158 t159 t160 t161 t162 t163 t164 t165 t166 t167 t168 t169 t170 t171 t172 t173 t174 t175 t176 t177 t178 t179 t180 t181 t182 t183 t184 t185 t186 t187 t188 t189 t190 t191 t192 t193 t194 t195 t196 t197 t198 t199 t200 t201 t202 t203 t204 t205 t206 t207 t208 t209 t210 t211 t212 t213 t214 t215 t216 t217 t218 t219 t220 t221 t222 t223 t224 t225 t226 t227 t228 t229 t230 t231 t232 t233 t234 t235 t236 t237 t238 t239 t240 t241 t242 t243 t244 t245 t246 t247 t248 t249 t250 t251 t252 t253 t254 t255 t256 t257 t258 t259 t260 t261 t262 t263 t264 t265 t266 t267 t268 t269 t270 t271 t272 t273 t274 t275 t276 t277 t278 t279 t280 t281 t282 t283 t284 t285 t286 t287 t288 t289 t290 t291 t292 t293 t294 t295 t296 t297 t298 t299 t300 t301 t302 t303 t304 t305 t306 t307 t308 t309 t310 t311 t312 t313 t314 t315 t316 t317 t318 t319 t320 t321 t322 t323 t324 t325 t326 t327 t328 t329 t330 t331 t332 t333 t334 t335 t336 t337 t338 t339 t340 t341 t342 t343 t344 t345 t346 t347 t348 t349 t350 t351 t352 t353 t354 t355 t356 t357 t358 t359 t360 t361 t362 t363 t364 t365 t366 t367 t368 t369 t370 t371 t372 t373 t374 t375 t376 t377 t378 t379 t380 t381 t382 t383 t384 t385 t386 t387 t388 t389 t390 t391 t392 t393 t394 t395 t396 t397 t398 t399 t400 t401 t402 t403 t404 t405 t406 t407 t408 t409 t410 t411 t412 t413 t414 t415 t416 t417 t418 t419 t420 t421 t422 t423 t424 t425 t426 t427 t428 t429 t430 t431 t432 t433 t434 t435 t436 t437 t438 t439 t440 t441 t442 t443 t444 t445 t446 t447 t448 t449 t450 t451 t452 t453 t454 t455 t456 t457 t458 t459 t460 t461 t462 t463 t464 t465 t466 t467 t468 t469 t470 t471 t472 t473 t474 t475 t476 t477 t478 t479 t480 t481 t482 t483 t484 t485 t486 t487 t488 t489 t490 t491 t492 t493 t494 t495 t496 t497 t498 t499 t500 t501 t502 t503 t504 t505 t506 t507 t508 t509 t510 t511 t512 t513 t514 t515 t516 t517 t518 t519 t520 t521 t522 t523 t524 t525 t526 t527 t528 t529 t530 t531 t532 t533 t534 t535 t536 t537 t538 t539 t540 t541 t542 t543 t544 t545 t546 t547 t548 t549 t550 t551 t552 t553 t554 t555 t556 t557 t558 t559 t560 t561 t562 t563 t564 t565 t566 t567 t568 t569 t570 t571 t572 t573 t574 t575 t576 t577 t578 t579 t580 t581 t582 t583 t584 t585 t586 t587 t588 t589 t590 t591 t592 t593 t594 t595 t596 t597 t598 t599 t600 t601 t602 t603 t604 t605 t606 t607 t608 t609 t610 t611 t612 t613 t614 t615 t616 t617 t618 t619 t620 t621 t622 t623 t624 t625 t626 t627 t628 t629 t630 t631 t632 t633 t634 t635 t636 t637 t638 t639 t640 t641 t642 t643 t644 t645 t646 t647 t648 t649 t650 t651 t652 t653 t654 t655 t656 t657 t658 t659 t660 t661 t662 t663 t664 t665 t666 t667 t668 t669 t670 t671 t672 t673 t674 t675 t676 t677 t678 t679 t680 t681 t682 t683 t684 t685 t686 t687 t688 t689 t690 t691 t692 t693 t694 t695 t696 t697 t698 t699 t700 t701 t702 t703 t704 t705 t706 t707 t708 t709 t710 t711 t712 t713 t714 t715 t716 t717 t718 t719 t720 t721 t722 t723 t724 t725 t726 t727 t728 t729 t730 t731 t732 t733 t734 t735 t736 t737 t738 t739 t740 t741 t742 t743 t744 t745 t746 t747 t748 t749 t750 t751 t752 t753 t754 t755 t756 t757 t758 t759 t760 t761 t762 t763 t764 t765 t766 t767 t768 t769 t770 t771 t772 t773 t774 t775 t776 t777 t778 t779 t780 t781 t782 t783 t784 t785 t786 t787 t788 t789 t790 t791 t792 t793 t794 t795 t796 t797 t798 t799 t800 t801 t802 t803 t804 t805 t806 t807 t808 t809 t810 t811 t812 t813 t814 t815 t816 t817 t818 t819 t820 t821 t822 t823 t824 t825 t826 t827 t828 t829 t830 t831 t832 t833 t834 t835 t836 t837 t838 t839 t840 t841 t842 t843 t844 t845 t846 t847 t848 t849 t850 t851 t852 t853 t854 t855 t856 t857 t858 t859 t860 t861 t862 t863 t864 t865 t866 t867 t868 t869 t870 t871 t872 t873 t874 t875 t876 t877 t878 t879 t880 t881 t882 t883 t884 t885 t886 t887 t888 t889 t890 t891 t892 t893 t894 t895 t896 t897 t898 t899 t900 t901 t902 t903 t904 t905 t906 t907 t908 t909 t910 t911 t912 t913 t914 t915 t916 t917 t918 t919 t920 t921 t922 t923 t924 t925 t926 t927 t928 t929 t930 t931 t932 t933 t934 t935 t936 t937 t938 t939 t940 t941 t942 t943 t944 t945 t946 t947 t948 t949 t950 t951 t952 t953 t954 t955 t956 t957 t958 t959 t960 t961 t962 t963 t964 t965 t966 t967 t968 t969 t970 t971 t972 t973 t974 t975 t976 t977 t978 t979 t980 t981 t982 t983 t984 t985 t986 t987 t988 t989 t990 t991 t992 t993 t994 t995 t996 t997 t998 t999 t1000 t1001 t1002 t1003 t1004 t1005 t1006 t1007 t1008 t1009 t1010 t1011 t1012 t1013 t1014 t1015 t1016 t1017 t1018 t1019 t1020 t1021 t1022 t1023 t1024 t1025 t1026 t1027 t1028 t1029 t1030 t1031 t1032 t1033 t1034 t1035 t1036 t1037 t1038 t1039 t1040 t1041 t1042 t1043 t1044 t1045 t1046 t1047 t1048 t1049 t1050 t1051 t1052 t1053 t1054 t1055 t1056 t1057 t1058 t1059 t1060 t1061 t1062 t1063 t1064 t1065 t1066 t1067 t1068 t1069 t1070 t1071 t1072 t1073 t1074 t1075 t1076 t1077 t1078 t1079 t1080 t1081 t1082 t1083 t1084 t1085 t1086 t1087 t1088 t1089 t1090 t1091 t1092 t1093 t1094 t1095 t1096 t1097 t1098 t1099 t1100 t1101 t1102 t1103 t1104 t1105 t1106 t1107 t1108 t1109 t1110 t1111 t1112 t1113 t1114 t1115 t1116 t1117 t1118 t1119 t1120 t1121 t1122 t1123 t1124 t1125 t1126 t1127 t1128 t1129 t1130 t1131 t1132 t1133 t1134 t1135 t1136 t1137 t1138 t1139 t1140 t1141 t1142 t1143 t1144 t1145 t1146 t1147 t1148 t1149 t1150 t1151 t1152 t1153 t1154 t1155 t1156 t1157 t1158 t1159 t1160 t1161 t1162 t1163 t1164 t1165 t1166 t1167 t1168 t1169 t1170 t1171 t1172 t1173 t1174 t1175 t1176 t1177 t1178 t1179 t1180 t1181 t1182 t1183 t1184 t1185 t1186 t1187 t1188 t1189 t1190 t1191 t1192 t1193 t1194 t1195 t1196 t1197 t1198 t1199 t1200 t1201 t1202 t1203 t1204 t1205 t1206 t1207 t1208 t1209 t1210 t1211 t1212 t1213 t1214 t1215 t1216 t1217 t1218 t1219 t1220 t1221 t1222 t1223 t1224 t1225 t1226 t1227 t1228 t1229 t1230 t1231 t1232 t1233 t1234 t1235 t1236 t1237 t1238 t1239 t1240 t1241 t1242 t1243 t1244 t1245 t1246 t1247 t1248 t1249 t1250 t1251 t1252 t1253 t1254 t1255 t1256 t1257 t1258 t1259 t1260 t1261 t1262 t1263 t1264 t1265 t1266 t1267 t1268 t1269 t1270 t1271 t1272 t1273 t1274 t1275 t1276 t1277 t1278 t1279 t1280 t1281 t1282 t1283 t1284 t1285 t1286 t1287 t1288 t1289 t1290 t1291 t1292 t1293 t1294 t1295 t1296 t1297 t1298 t1299 t1300 t1301 t1302 t1303 t1304 t1305 t1306 t1307 t1308 t1309 t1310 t1311 t1312 t1313 t1314 t1315 t1316 t1317 t1318 t1319 t1320 t1321 t1322 t1323 t1324 t1325 t1326 t1327 t1328 t1329 t1330 t1331 t1332 t1333 t1334 t1335 t1336 t1337 t1338 t1339 t1340 t1341 t1342 t1343 t1344 t1345 t1346 t1347 t1348 t1349 t1350 t1351 t1352 t1353 t1354 t1355 t1356 t1357 t1358 t1359 t1360 t1361 t1362 t1363 t1364 t1365 t1366 t1367 t1368 t1369 t1370 t1371 t1372 t1373 t1374 t1375 t1376 t1377 t1378 t1379 t1380 t1381 t1382 t1383 t1384 t1385 t1386 t1387 t1388 t1389 t1390 t1391 t1392 t1393 t1394 t1395 t1396 t1397 t1398 t1399 t1400 t1401 t1402 t1403 t1404 t1405 t1406 t1407 t1408 t1409 t1410 t1411 t1412 t1413 t1414 t1415 t1416 t1417 t1418 t1419 t1420 t1421 t1422 t1423 t1424 t1425 t1426 t1427 t1428 t1429 t1430 t1431 t1432 t1433 t1434 t1435 t1436 t1437 t1438 t1439 t1440
.PHONY:t1
t1:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls2_maxit100_0.log
.PHONY:t2
t2:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls2_maxit100_1.log
.PHONY:t3
t3:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls2_maxit100_2.log
.PHONY:t4
t4:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls2_maxit100_3.log
.PHONY:t5
t5:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls2_maxit100_4.log
.PHONY:t6
t6:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls2_maxit100_5.log
.PHONY:t7
t7:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls2_maxit100_6.log
.PHONY:t8
t8:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls2_maxit100_7.log
.PHONY:t9
t9:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls2_maxit100_8.log
.PHONY:t10
t10:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls2_maxit100_9.log
.PHONY:t11
t11:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls5_maxit100_0.log
.PHONY:t12
t12:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls5_maxit100_1.log
.PHONY:t13
t13:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls5_maxit100_2.log
.PHONY:t14
t14:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls5_maxit100_3.log
.PHONY:t15
t15:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls5_maxit100_4.log
.PHONY:t16
t16:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls5_maxit100_5.log
.PHONY:t17
t17:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls5_maxit100_6.log
.PHONY:t18
t18:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls5_maxit100_7.log
.PHONY:t19
t19:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls5_maxit100_8.log
.PHONY:t20
t20:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls5_maxit100_9.log
.PHONY:t21
t21:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls10_maxit100_0.log
.PHONY:t22
t22:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls10_maxit100_1.log
.PHONY:t23
t23:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls10_maxit100_2.log
.PHONY:t24
t24:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls10_maxit100_3.log
.PHONY:t25
t25:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls10_maxit100_4.log
.PHONY:t26
t26:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls10_maxit100_5.log
.PHONY:t27
t27:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls10_maxit100_6.log
.PHONY:t28
t28:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls10_maxit100_7.log
.PHONY:t29
t29:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls10_maxit100_8.log
.PHONY:t30
t30:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls10_maxit100_9.log
.PHONY:t31
t31:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls30_maxit100_0.log
.PHONY:t32
t32:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls30_maxit100_1.log
.PHONY:t33
t33:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls30_maxit100_2.log
.PHONY:t34
t34:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls30_maxit100_3.log
.PHONY:t35
t35:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls30_maxit100_4.log
.PHONY:t36
t36:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls30_maxit100_5.log
.PHONY:t37
t37:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls30_maxit100_6.log
.PHONY:t38
t38:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls30_maxit100_7.log
.PHONY:t39
t39:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls30_maxit100_8.log
.PHONY:t40
t40:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls30_maxit100_9.log
.PHONY:t41
t41:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls50_maxit100_0.log
.PHONY:t42
t42:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls50_maxit100_1.log
.PHONY:t43
t43:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls50_maxit100_2.log
.PHONY:t44
t44:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls50_maxit100_3.log
.PHONY:t45
t45:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls50_maxit100_4.log
.PHONY:t46
t46:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls50_maxit100_5.log
.PHONY:t47
t47:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls50_maxit100_6.log
.PHONY:t48
t48:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls50_maxit100_7.log
.PHONY:t49
t49:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls50_maxit100_8.log
.PHONY:t50
t50:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls50_maxit100_9.log
.PHONY:t51
t51:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls80_maxit100_0.log
.PHONY:t52
t52:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls80_maxit100_1.log
.PHONY:t53
t53:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls80_maxit100_2.log
.PHONY:t54
t54:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls80_maxit100_3.log
.PHONY:t55
t55:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls80_maxit100_4.log
.PHONY:t56
t56:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls80_maxit100_5.log
.PHONY:t57
t57:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls80_maxit100_6.log
.PHONY:t58
t58:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls80_maxit100_7.log
.PHONY:t59
t59:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls80_maxit100_8.log
.PHONY:t60
t60:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls80_maxit100_9.log
.PHONY:t61
t61:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls100_maxit100_0.log
.PHONY:t62
t62:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls100_maxit100_1.log
.PHONY:t63
t63:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls100_maxit100_2.log
.PHONY:t64
t64:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls100_maxit100_3.log
.PHONY:t65
t65:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls100_maxit100_4.log
.PHONY:t66
t66:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls100_maxit100_5.log
.PHONY:t67
t67:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls100_maxit100_6.log
.PHONY:t68
t68:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls100_maxit100_7.log
.PHONY:t69
t69:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls100_maxit100_8.log
.PHONY:t70
t70:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls100_maxit100_9.log
.PHONY:t71
t71:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls150_maxit100_0.log
.PHONY:t72
t72:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls150_maxit100_1.log
.PHONY:t73
t73:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls150_maxit100_2.log
.PHONY:t74
t74:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls150_maxit100_3.log
.PHONY:t75
t75:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls150_maxit100_4.log
.PHONY:t76
t76:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls150_maxit100_5.log
.PHONY:t77
t77:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls150_maxit100_6.log
.PHONY:t78
t78:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls150_maxit100_7.log
.PHONY:t79
t79:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls150_maxit100_8.log
.PHONY:t80
t80:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls150_maxit100_9.log
.PHONY:t81
t81:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls200_maxit100_0.log
.PHONY:t82
t82:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls200_maxit100_1.log
.PHONY:t83
t83:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls200_maxit100_2.log
.PHONY:t84
t84:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls200_maxit100_3.log
.PHONY:t85
t85:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls200_maxit100_4.log
.PHONY:t86
t86:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls200_maxit100_5.log
.PHONY:t87
t87:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls200_maxit100_6.log
.PHONY:t88
t88:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls200_maxit100_7.log
.PHONY:t89
t89:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls200_maxit100_8.log
.PHONY:t90
t90:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.1_hls200_maxit100_9.log
.PHONY:t91
t91:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls2_maxit100_0.log
.PHONY:t92
t92:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls2_maxit100_1.log
.PHONY:t93
t93:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls2_maxit100_2.log
.PHONY:t94
t94:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls2_maxit100_3.log
.PHONY:t95
t95:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls2_maxit100_4.log
.PHONY:t96
t96:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls2_maxit100_5.log
.PHONY:t97
t97:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls2_maxit100_6.log
.PHONY:t98
t98:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls2_maxit100_7.log
.PHONY:t99
t99:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls2_maxit100_8.log
.PHONY:t100
t100:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls2_maxit100_9.log
.PHONY:t101
t101:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls5_maxit100_0.log
.PHONY:t102
t102:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls5_maxit100_1.log
.PHONY:t103
t103:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls5_maxit100_2.log
.PHONY:t104
t104:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls5_maxit100_3.log
.PHONY:t105
t105:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls5_maxit100_4.log
.PHONY:t106
t106:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls5_maxit100_5.log
.PHONY:t107
t107:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls5_maxit100_6.log
.PHONY:t108
t108:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls5_maxit100_7.log
.PHONY:t109
t109:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls5_maxit100_8.log
.PHONY:t110
t110:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls5_maxit100_9.log
.PHONY:t111
t111:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls10_maxit100_0.log
.PHONY:t112
t112:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls10_maxit100_1.log
.PHONY:t113
t113:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls10_maxit100_2.log
.PHONY:t114
t114:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls10_maxit100_3.log
.PHONY:t115
t115:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls10_maxit100_4.log
.PHONY:t116
t116:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls10_maxit100_5.log
.PHONY:t117
t117:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls10_maxit100_6.log
.PHONY:t118
t118:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls10_maxit100_7.log
.PHONY:t119
t119:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls10_maxit100_8.log
.PHONY:t120
t120:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls10_maxit100_9.log
.PHONY:t121
t121:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls30_maxit100_0.log
.PHONY:t122
t122:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls30_maxit100_1.log
.PHONY:t123
t123:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls30_maxit100_2.log
.PHONY:t124
t124:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls30_maxit100_3.log
.PHONY:t125
t125:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls30_maxit100_4.log
.PHONY:t126
t126:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls30_maxit100_5.log
.PHONY:t127
t127:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls30_maxit100_6.log
.PHONY:t128
t128:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls30_maxit100_7.log
.PHONY:t129
t129:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls30_maxit100_8.log
.PHONY:t130
t130:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls30_maxit100_9.log
.PHONY:t131
t131:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls50_maxit100_0.log
.PHONY:t132
t132:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls50_maxit100_1.log
.PHONY:t133
t133:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls50_maxit100_2.log
.PHONY:t134
t134:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls50_maxit100_3.log
.PHONY:t135
t135:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls50_maxit100_4.log
.PHONY:t136
t136:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls50_maxit100_5.log
.PHONY:t137
t137:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls50_maxit100_6.log
.PHONY:t138
t138:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls50_maxit100_7.log
.PHONY:t139
t139:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls50_maxit100_8.log
.PHONY:t140
t140:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls50_maxit100_9.log
.PHONY:t141
t141:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls80_maxit100_0.log
.PHONY:t142
t142:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls80_maxit100_1.log
.PHONY:t143
t143:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls80_maxit100_2.log
.PHONY:t144
t144:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls80_maxit100_3.log
.PHONY:t145
t145:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls80_maxit100_4.log
.PHONY:t146
t146:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls80_maxit100_5.log
.PHONY:t147
t147:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls80_maxit100_6.log
.PHONY:t148
t148:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls80_maxit100_7.log
.PHONY:t149
t149:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls80_maxit100_8.log
.PHONY:t150
t150:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls80_maxit100_9.log
.PHONY:t151
t151:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls100_maxit100_0.log
.PHONY:t152
t152:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls100_maxit100_1.log
.PHONY:t153
t153:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls100_maxit100_2.log
.PHONY:t154
t154:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls100_maxit100_3.log
.PHONY:t155
t155:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls100_maxit100_4.log
.PHONY:t156
t156:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls100_maxit100_5.log
.PHONY:t157
t157:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls100_maxit100_6.log
.PHONY:t158
t158:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls100_maxit100_7.log
.PHONY:t159
t159:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls100_maxit100_8.log
.PHONY:t160
t160:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls100_maxit100_9.log
.PHONY:t161
t161:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls150_maxit100_0.log
.PHONY:t162
t162:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls150_maxit100_1.log
.PHONY:t163
t163:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls150_maxit100_2.log
.PHONY:t164
t164:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls150_maxit100_3.log
.PHONY:t165
t165:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls150_maxit100_4.log
.PHONY:t166
t166:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls150_maxit100_5.log
.PHONY:t167
t167:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls150_maxit100_6.log
.PHONY:t168
t168:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls150_maxit100_7.log
.PHONY:t169
t169:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls150_maxit100_8.log
.PHONY:t170
t170:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls150_maxit100_9.log
.PHONY:t171
t171:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls200_maxit100_0.log
.PHONY:t172
t172:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls200_maxit100_1.log
.PHONY:t173
t173:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls200_maxit100_2.log
.PHONY:t174
t174:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls200_maxit100_3.log
.PHONY:t175
t175:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls200_maxit100_4.log
.PHONY:t176
t176:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls200_maxit100_5.log
.PHONY:t177
t177:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls200_maxit100_6.log
.PHONY:t178
t178:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls200_maxit100_7.log
.PHONY:t179
t179:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls200_maxit100_8.log
.PHONY:t180
t180:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_uniform_vp0.2_hls200_maxit100_9.log
.PHONY:t181
t181:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls2_maxit100_0.log
.PHONY:t182
t182:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls2_maxit100_1.log
.PHONY:t183
t183:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls2_maxit100_2.log
.PHONY:t184
t184:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls2_maxit100_3.log
.PHONY:t185
t185:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls2_maxit100_4.log
.PHONY:t186
t186:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls2_maxit100_5.log
.PHONY:t187
t187:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls2_maxit100_6.log
.PHONY:t188
t188:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls2_maxit100_7.log
.PHONY:t189
t189:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls2_maxit100_8.log
.PHONY:t190
t190:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls2_maxit100_9.log
.PHONY:t191
t191:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls5_maxit100_0.log
.PHONY:t192
t192:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls5_maxit100_1.log
.PHONY:t193
t193:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls5_maxit100_2.log
.PHONY:t194
t194:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls5_maxit100_3.log
.PHONY:t195
t195:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls5_maxit100_4.log
.PHONY:t196
t196:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls5_maxit100_5.log
.PHONY:t197
t197:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls5_maxit100_6.log
.PHONY:t198
t198:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls5_maxit100_7.log
.PHONY:t199
t199:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls5_maxit100_8.log
.PHONY:t200
t200:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls5_maxit100_9.log
.PHONY:t201
t201:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls10_maxit100_0.log
.PHONY:t202
t202:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls10_maxit100_1.log
.PHONY:t203
t203:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls10_maxit100_2.log
.PHONY:t204
t204:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls10_maxit100_3.log
.PHONY:t205
t205:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls10_maxit100_4.log
.PHONY:t206
t206:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls10_maxit100_5.log
.PHONY:t207
t207:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls10_maxit100_6.log
.PHONY:t208
t208:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls10_maxit100_7.log
.PHONY:t209
t209:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls10_maxit100_8.log
.PHONY:t210
t210:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls10_maxit100_9.log
.PHONY:t211
t211:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls30_maxit100_0.log
.PHONY:t212
t212:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls30_maxit100_1.log
.PHONY:t213
t213:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls30_maxit100_2.log
.PHONY:t214
t214:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls30_maxit100_3.log
.PHONY:t215
t215:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls30_maxit100_4.log
.PHONY:t216
t216:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls30_maxit100_5.log
.PHONY:t217
t217:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls30_maxit100_6.log
.PHONY:t218
t218:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls30_maxit100_7.log
.PHONY:t219
t219:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls30_maxit100_8.log
.PHONY:t220
t220:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls30_maxit100_9.log
.PHONY:t221
t221:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls50_maxit100_0.log
.PHONY:t222
t222:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls50_maxit100_1.log
.PHONY:t223
t223:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls50_maxit100_2.log
.PHONY:t224
t224:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls50_maxit100_3.log
.PHONY:t225
t225:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls50_maxit100_4.log
.PHONY:t226
t226:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls50_maxit100_5.log
.PHONY:t227
t227:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls50_maxit100_6.log
.PHONY:t228
t228:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls50_maxit100_7.log
.PHONY:t229
t229:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls50_maxit100_8.log
.PHONY:t230
t230:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls50_maxit100_9.log
.PHONY:t231
t231:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls80_maxit100_0.log
.PHONY:t232
t232:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls80_maxit100_1.log
.PHONY:t233
t233:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls80_maxit100_2.log
.PHONY:t234
t234:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls80_maxit100_3.log
.PHONY:t235
t235:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls80_maxit100_4.log
.PHONY:t236
t236:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls80_maxit100_5.log
.PHONY:t237
t237:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls80_maxit100_6.log
.PHONY:t238
t238:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls80_maxit100_7.log
.PHONY:t239
t239:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls80_maxit100_8.log
.PHONY:t240
t240:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls80_maxit100_9.log
.PHONY:t241
t241:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls100_maxit100_0.log
.PHONY:t242
t242:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls100_maxit100_1.log
.PHONY:t243
t243:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls100_maxit100_2.log
.PHONY:t244
t244:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls100_maxit100_3.log
.PHONY:t245
t245:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls100_maxit100_4.log
.PHONY:t246
t246:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls100_maxit100_5.log
.PHONY:t247
t247:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls100_maxit100_6.log
.PHONY:t248
t248:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls100_maxit100_7.log
.PHONY:t249
t249:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls100_maxit100_8.log
.PHONY:t250
t250:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls100_maxit100_9.log
.PHONY:t251
t251:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls150_maxit100_0.log
.PHONY:t252
t252:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls150_maxit100_1.log
.PHONY:t253
t253:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls150_maxit100_2.log
.PHONY:t254
t254:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls150_maxit100_3.log
.PHONY:t255
t255:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls150_maxit100_4.log
.PHONY:t256
t256:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls150_maxit100_5.log
.PHONY:t257
t257:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls150_maxit100_6.log
.PHONY:t258
t258:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls150_maxit100_7.log
.PHONY:t259
t259:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls150_maxit100_8.log
.PHONY:t260
t260:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls150_maxit100_9.log
.PHONY:t261
t261:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls200_maxit100_0.log
.PHONY:t262
t262:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls200_maxit100_1.log
.PHONY:t263
t263:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls200_maxit100_2.log
.PHONY:t264
t264:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls200_maxit100_3.log
.PHONY:t265
t265:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls200_maxit100_4.log
.PHONY:t266
t266:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls200_maxit100_5.log
.PHONY:t267
t267:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls200_maxit100_6.log
.PHONY:t268
t268:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls200_maxit100_7.log
.PHONY:t269
t269:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls200_maxit100_8.log
.PHONY:t270
t270:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.1_hls200_maxit100_9.log
.PHONY:t271
t271:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls2_maxit100_0.log
.PHONY:t272
t272:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls2_maxit100_1.log
.PHONY:t273
t273:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls2_maxit100_2.log
.PHONY:t274
t274:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls2_maxit100_3.log
.PHONY:t275
t275:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls2_maxit100_4.log
.PHONY:t276
t276:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls2_maxit100_5.log
.PHONY:t277
t277:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls2_maxit100_6.log
.PHONY:t278
t278:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls2_maxit100_7.log
.PHONY:t279
t279:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls2_maxit100_8.log
.PHONY:t280
t280:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls2_maxit100_9.log
.PHONY:t281
t281:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls5_maxit100_0.log
.PHONY:t282
t282:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls5_maxit100_1.log
.PHONY:t283
t283:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls5_maxit100_2.log
.PHONY:t284
t284:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls5_maxit100_3.log
.PHONY:t285
t285:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls5_maxit100_4.log
.PHONY:t286
t286:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls5_maxit100_5.log
.PHONY:t287
t287:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls5_maxit100_6.log
.PHONY:t288
t288:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls5_maxit100_7.log
.PHONY:t289
t289:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls5_maxit100_8.log
.PHONY:t290
t290:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls5_maxit100_9.log
.PHONY:t291
t291:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls10_maxit100_0.log
.PHONY:t292
t292:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls10_maxit100_1.log
.PHONY:t293
t293:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls10_maxit100_2.log
.PHONY:t294
t294:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls10_maxit100_3.log
.PHONY:t295
t295:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls10_maxit100_4.log
.PHONY:t296
t296:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls10_maxit100_5.log
.PHONY:t297
t297:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls10_maxit100_6.log
.PHONY:t298
t298:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls10_maxit100_7.log
.PHONY:t299
t299:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls10_maxit100_8.log
.PHONY:t300
t300:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls10_maxit100_9.log
.PHONY:t301
t301:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls30_maxit100_0.log
.PHONY:t302
t302:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls30_maxit100_1.log
.PHONY:t303
t303:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls30_maxit100_2.log
.PHONY:t304
t304:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls30_maxit100_3.log
.PHONY:t305
t305:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls30_maxit100_4.log
.PHONY:t306
t306:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls30_maxit100_5.log
.PHONY:t307
t307:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls30_maxit100_6.log
.PHONY:t308
t308:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls30_maxit100_7.log
.PHONY:t309
t309:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls30_maxit100_8.log
.PHONY:t310
t310:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls30_maxit100_9.log
.PHONY:t311
t311:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls50_maxit100_0.log
.PHONY:t312
t312:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls50_maxit100_1.log
.PHONY:t313
t313:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls50_maxit100_2.log
.PHONY:t314
t314:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls50_maxit100_3.log
.PHONY:t315
t315:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls50_maxit100_4.log
.PHONY:t316
t316:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls50_maxit100_5.log
.PHONY:t317
t317:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls50_maxit100_6.log
.PHONY:t318
t318:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls50_maxit100_7.log
.PHONY:t319
t319:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls50_maxit100_8.log
.PHONY:t320
t320:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls50_maxit100_9.log
.PHONY:t321
t321:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls80_maxit100_0.log
.PHONY:t322
t322:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls80_maxit100_1.log
.PHONY:t323
t323:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls80_maxit100_2.log
.PHONY:t324
t324:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls80_maxit100_3.log
.PHONY:t325
t325:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls80_maxit100_4.log
.PHONY:t326
t326:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls80_maxit100_5.log
.PHONY:t327
t327:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls80_maxit100_6.log
.PHONY:t328
t328:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls80_maxit100_7.log
.PHONY:t329
t329:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls80_maxit100_8.log
.PHONY:t330
t330:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls80_maxit100_9.log
.PHONY:t331
t331:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls100_maxit100_0.log
.PHONY:t332
t332:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls100_maxit100_1.log
.PHONY:t333
t333:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls100_maxit100_2.log
.PHONY:t334
t334:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls100_maxit100_3.log
.PHONY:t335
t335:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls100_maxit100_4.log
.PHONY:t336
t336:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls100_maxit100_5.log
.PHONY:t337
t337:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls100_maxit100_6.log
.PHONY:t338
t338:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls100_maxit100_7.log
.PHONY:t339
t339:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls100_maxit100_8.log
.PHONY:t340
t340:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls100_maxit100_9.log
.PHONY:t341
t341:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls150_maxit100_0.log
.PHONY:t342
t342:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls150_maxit100_1.log
.PHONY:t343
t343:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls150_maxit100_2.log
.PHONY:t344
t344:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls150_maxit100_3.log
.PHONY:t345
t345:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls150_maxit100_4.log
.PHONY:t346
t346:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls150_maxit100_5.log
.PHONY:t347
t347:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls150_maxit100_6.log
.PHONY:t348
t348:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls150_maxit100_7.log
.PHONY:t349
t349:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls150_maxit100_8.log
.PHONY:t350
t350:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls150_maxit100_9.log
.PHONY:t351
t351:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls200_maxit100_0.log
.PHONY:t352
t352:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls200_maxit100_1.log
.PHONY:t353
t353:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls200_maxit100_2.log
.PHONY:t354
t354:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls200_maxit100_3.log
.PHONY:t355
t355:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls200_maxit100_4.log
.PHONY:t356
t356:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls200_maxit100_5.log
.PHONY:t357
t357:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls200_maxit100_6.log
.PHONY:t358
t358:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls200_maxit100_7.log
.PHONY:t359
t359:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls200_maxit100_8.log
.PHONY:t360
t360:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_glorot_normal_vp0.2_hls200_maxit100_9.log
.PHONY:t361
t361:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls2_maxit100_0.log
.PHONY:t362
t362:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls2_maxit100_1.log
.PHONY:t363
t363:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls2_maxit100_2.log
.PHONY:t364
t364:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls2_maxit100_3.log
.PHONY:t365
t365:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls2_maxit100_4.log
.PHONY:t366
t366:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls2_maxit100_5.log
.PHONY:t367
t367:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls2_maxit100_6.log
.PHONY:t368
t368:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls2_maxit100_7.log
.PHONY:t369
t369:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls2_maxit100_8.log
.PHONY:t370
t370:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls2_maxit100_9.log
.PHONY:t371
t371:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls5_maxit100_0.log
.PHONY:t372
t372:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls5_maxit100_1.log
.PHONY:t373
t373:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls5_maxit100_2.log
.PHONY:t374
t374:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls5_maxit100_3.log
.PHONY:t375
t375:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls5_maxit100_4.log
.PHONY:t376
t376:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls5_maxit100_5.log
.PHONY:t377
t377:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls5_maxit100_6.log
.PHONY:t378
t378:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls5_maxit100_7.log
.PHONY:t379
t379:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls5_maxit100_8.log
.PHONY:t380
t380:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls5_maxit100_9.log
.PHONY:t381
t381:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls10_maxit100_0.log
.PHONY:t382
t382:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls10_maxit100_1.log
.PHONY:t383
t383:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls10_maxit100_2.log
.PHONY:t384
t384:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls10_maxit100_3.log
.PHONY:t385
t385:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls10_maxit100_4.log
.PHONY:t386
t386:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls10_maxit100_5.log
.PHONY:t387
t387:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls10_maxit100_6.log
.PHONY:t388
t388:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls10_maxit100_7.log
.PHONY:t389
t389:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls10_maxit100_8.log
.PHONY:t390
t390:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls10_maxit100_9.log
.PHONY:t391
t391:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls30_maxit100_0.log
.PHONY:t392
t392:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls30_maxit100_1.log
.PHONY:t393
t393:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls30_maxit100_2.log
.PHONY:t394
t394:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls30_maxit100_3.log
.PHONY:t395
t395:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls30_maxit100_4.log
.PHONY:t396
t396:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls30_maxit100_5.log
.PHONY:t397
t397:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls30_maxit100_6.log
.PHONY:t398
t398:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls30_maxit100_7.log
.PHONY:t399
t399:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls30_maxit100_8.log
.PHONY:t400
t400:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls30_maxit100_9.log
.PHONY:t401
t401:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls50_maxit100_0.log
.PHONY:t402
t402:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls50_maxit100_1.log
.PHONY:t403
t403:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls50_maxit100_2.log
.PHONY:t404
t404:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls50_maxit100_3.log
.PHONY:t405
t405:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls50_maxit100_4.log
.PHONY:t406
t406:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls50_maxit100_5.log
.PHONY:t407
t407:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls50_maxit100_6.log
.PHONY:t408
t408:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls50_maxit100_7.log
.PHONY:t409
t409:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls50_maxit100_8.log
.PHONY:t410
t410:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls50_maxit100_9.log
.PHONY:t411
t411:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls80_maxit100_0.log
.PHONY:t412
t412:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls80_maxit100_1.log
.PHONY:t413
t413:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls80_maxit100_2.log
.PHONY:t414
t414:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls80_maxit100_3.log
.PHONY:t415
t415:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls80_maxit100_4.log
.PHONY:t416
t416:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls80_maxit100_5.log
.PHONY:t417
t417:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls80_maxit100_6.log
.PHONY:t418
t418:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls80_maxit100_7.log
.PHONY:t419
t419:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls80_maxit100_8.log
.PHONY:t420
t420:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls80_maxit100_9.log
.PHONY:t421
t421:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls100_maxit100_0.log
.PHONY:t422
t422:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls100_maxit100_1.log
.PHONY:t423
t423:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls100_maxit100_2.log
.PHONY:t424
t424:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls100_maxit100_3.log
.PHONY:t425
t425:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls100_maxit100_4.log
.PHONY:t426
t426:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls100_maxit100_5.log
.PHONY:t427
t427:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls100_maxit100_6.log
.PHONY:t428
t428:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls100_maxit100_7.log
.PHONY:t429
t429:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls100_maxit100_8.log
.PHONY:t430
t430:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls100_maxit100_9.log
.PHONY:t431
t431:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls150_maxit100_0.log
.PHONY:t432
t432:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls150_maxit100_1.log
.PHONY:t433
t433:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls150_maxit100_2.log
.PHONY:t434
t434:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls150_maxit100_3.log
.PHONY:t435
t435:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls150_maxit100_4.log
.PHONY:t436
t436:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls150_maxit100_5.log
.PHONY:t437
t437:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls150_maxit100_6.log
.PHONY:t438
t438:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls150_maxit100_7.log
.PHONY:t439
t439:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls150_maxit100_8.log
.PHONY:t440
t440:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls150_maxit100_9.log
.PHONY:t441
t441:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls200_maxit100_0.log
.PHONY:t442
t442:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls200_maxit100_1.log
.PHONY:t443
t443:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls200_maxit100_2.log
.PHONY:t444
t444:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls200_maxit100_3.log
.PHONY:t445
t445:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls200_maxit100_4.log
.PHONY:t446
t446:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls200_maxit100_5.log
.PHONY:t447
t447:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls200_maxit100_6.log
.PHONY:t448
t448:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls200_maxit100_7.log
.PHONY:t449
t449:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls200_maxit100_8.log
.PHONY:t450
t450:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.1_hls200_maxit100_9.log
.PHONY:t451
t451:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls2_maxit100_0.log
.PHONY:t452
t452:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls2_maxit100_1.log
.PHONY:t453
t453:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls2_maxit100_2.log
.PHONY:t454
t454:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls2_maxit100_3.log
.PHONY:t455
t455:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls2_maxit100_4.log
.PHONY:t456
t456:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls2_maxit100_5.log
.PHONY:t457
t457:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls2_maxit100_6.log
.PHONY:t458
t458:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls2_maxit100_7.log
.PHONY:t459
t459:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls2_maxit100_8.log
.PHONY:t460
t460:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls2_maxit100_9.log
.PHONY:t461
t461:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls5_maxit100_0.log
.PHONY:t462
t462:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls5_maxit100_1.log
.PHONY:t463
t463:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls5_maxit100_2.log
.PHONY:t464
t464:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls5_maxit100_3.log
.PHONY:t465
t465:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls5_maxit100_4.log
.PHONY:t466
t466:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls5_maxit100_5.log
.PHONY:t467
t467:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls5_maxit100_6.log
.PHONY:t468
t468:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls5_maxit100_7.log
.PHONY:t469
t469:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls5_maxit100_8.log
.PHONY:t470
t470:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls5_maxit100_9.log
.PHONY:t471
t471:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls10_maxit100_0.log
.PHONY:t472
t472:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls10_maxit100_1.log
.PHONY:t473
t473:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls10_maxit100_2.log
.PHONY:t474
t474:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls10_maxit100_3.log
.PHONY:t475
t475:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls10_maxit100_4.log
.PHONY:t476
t476:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls10_maxit100_5.log
.PHONY:t477
t477:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls10_maxit100_6.log
.PHONY:t478
t478:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls10_maxit100_7.log
.PHONY:t479
t479:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls10_maxit100_8.log
.PHONY:t480
t480:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls10_maxit100_9.log
.PHONY:t481
t481:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls30_maxit100_0.log
.PHONY:t482
t482:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls30_maxit100_1.log
.PHONY:t483
t483:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls30_maxit100_2.log
.PHONY:t484
t484:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls30_maxit100_3.log
.PHONY:t485
t485:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls30_maxit100_4.log
.PHONY:t486
t486:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls30_maxit100_5.log
.PHONY:t487
t487:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls30_maxit100_6.log
.PHONY:t488
t488:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls30_maxit100_7.log
.PHONY:t489
t489:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls30_maxit100_8.log
.PHONY:t490
t490:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls30_maxit100_9.log
.PHONY:t491
t491:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls50_maxit100_0.log
.PHONY:t492
t492:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls50_maxit100_1.log
.PHONY:t493
t493:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls50_maxit100_2.log
.PHONY:t494
t494:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls50_maxit100_3.log
.PHONY:t495
t495:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls50_maxit100_4.log
.PHONY:t496
t496:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls50_maxit100_5.log
.PHONY:t497
t497:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls50_maxit100_6.log
.PHONY:t498
t498:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls50_maxit100_7.log
.PHONY:t499
t499:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls50_maxit100_8.log
.PHONY:t500
t500:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls50_maxit100_9.log
.PHONY:t501
t501:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls80_maxit100_0.log
.PHONY:t502
t502:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls80_maxit100_1.log
.PHONY:t503
t503:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls80_maxit100_2.log
.PHONY:t504
t504:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls80_maxit100_3.log
.PHONY:t505
t505:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls80_maxit100_4.log
.PHONY:t506
t506:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls80_maxit100_5.log
.PHONY:t507
t507:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls80_maxit100_6.log
.PHONY:t508
t508:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls80_maxit100_7.log
.PHONY:t509
t509:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls80_maxit100_8.log
.PHONY:t510
t510:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls80_maxit100_9.log
.PHONY:t511
t511:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls100_maxit100_0.log
.PHONY:t512
t512:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls100_maxit100_1.log
.PHONY:t513
t513:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls100_maxit100_2.log
.PHONY:t514
t514:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls100_maxit100_3.log
.PHONY:t515
t515:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls100_maxit100_4.log
.PHONY:t516
t516:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls100_maxit100_5.log
.PHONY:t517
t517:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls100_maxit100_6.log
.PHONY:t518
t518:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls100_maxit100_7.log
.PHONY:t519
t519:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls100_maxit100_8.log
.PHONY:t520
t520:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls100_maxit100_9.log
.PHONY:t521
t521:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls150_maxit100_0.log
.PHONY:t522
t522:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls150_maxit100_1.log
.PHONY:t523
t523:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls150_maxit100_2.log
.PHONY:t524
t524:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls150_maxit100_3.log
.PHONY:t525
t525:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls150_maxit100_4.log
.PHONY:t526
t526:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls150_maxit100_5.log
.PHONY:t527
t527:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls150_maxit100_6.log
.PHONY:t528
t528:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls150_maxit100_7.log
.PHONY:t529
t529:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls150_maxit100_8.log
.PHONY:t530
t530:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls150_maxit100_9.log
.PHONY:t531
t531:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls200_maxit100_0.log
.PHONY:t532
t532:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls200_maxit100_1.log
.PHONY:t533
t533:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls200_maxit100_2.log
.PHONY:t534
t534:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls200_maxit100_3.log
.PHONY:t535
t535:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls200_maxit100_4.log
.PHONY:t536
t536:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls200_maxit100_5.log
.PHONY:t537
t537:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls200_maxit100_6.log
.PHONY:t538
t538:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls200_maxit100_7.log
.PHONY:t539
t539:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls200_maxit100_8.log
.PHONY:t540
t540:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_normal_vp0.2_hls200_maxit100_9.log
.PHONY:t541
t541:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls2_maxit100_0.log
.PHONY:t542
t542:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls2_maxit100_1.log
.PHONY:t543
t543:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls2_maxit100_2.log
.PHONY:t544
t544:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls2_maxit100_3.log
.PHONY:t545
t545:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls2_maxit100_4.log
.PHONY:t546
t546:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls2_maxit100_5.log
.PHONY:t547
t547:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls2_maxit100_6.log
.PHONY:t548
t548:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls2_maxit100_7.log
.PHONY:t549
t549:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls2_maxit100_8.log
.PHONY:t550
t550:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls2_maxit100_9.log
.PHONY:t551
t551:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls5_maxit100_0.log
.PHONY:t552
t552:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls5_maxit100_1.log
.PHONY:t553
t553:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls5_maxit100_2.log
.PHONY:t554
t554:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls5_maxit100_3.log
.PHONY:t555
t555:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls5_maxit100_4.log
.PHONY:t556
t556:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls5_maxit100_5.log
.PHONY:t557
t557:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls5_maxit100_6.log
.PHONY:t558
t558:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls5_maxit100_7.log
.PHONY:t559
t559:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls5_maxit100_8.log
.PHONY:t560
t560:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls5_maxit100_9.log
.PHONY:t561
t561:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls10_maxit100_0.log
.PHONY:t562
t562:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls10_maxit100_1.log
.PHONY:t563
t563:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls10_maxit100_2.log
.PHONY:t564
t564:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls10_maxit100_3.log
.PHONY:t565
t565:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls10_maxit100_4.log
.PHONY:t566
t566:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls10_maxit100_5.log
.PHONY:t567
t567:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls10_maxit100_6.log
.PHONY:t568
t568:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls10_maxit100_7.log
.PHONY:t569
t569:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls10_maxit100_8.log
.PHONY:t570
t570:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls10_maxit100_9.log
.PHONY:t571
t571:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls30_maxit100_0.log
.PHONY:t572
t572:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls30_maxit100_1.log
.PHONY:t573
t573:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls30_maxit100_2.log
.PHONY:t574
t574:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls30_maxit100_3.log
.PHONY:t575
t575:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls30_maxit100_4.log
.PHONY:t576
t576:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls30_maxit100_5.log
.PHONY:t577
t577:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls30_maxit100_6.log
.PHONY:t578
t578:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls30_maxit100_7.log
.PHONY:t579
t579:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls30_maxit100_8.log
.PHONY:t580
t580:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls30_maxit100_9.log
.PHONY:t581
t581:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls50_maxit100_0.log
.PHONY:t582
t582:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls50_maxit100_1.log
.PHONY:t583
t583:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls50_maxit100_2.log
.PHONY:t584
t584:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls50_maxit100_3.log
.PHONY:t585
t585:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls50_maxit100_4.log
.PHONY:t586
t586:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls50_maxit100_5.log
.PHONY:t587
t587:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls50_maxit100_6.log
.PHONY:t588
t588:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls50_maxit100_7.log
.PHONY:t589
t589:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls50_maxit100_8.log
.PHONY:t590
t590:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls50_maxit100_9.log
.PHONY:t591
t591:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls80_maxit100_0.log
.PHONY:t592
t592:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls80_maxit100_1.log
.PHONY:t593
t593:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls80_maxit100_2.log
.PHONY:t594
t594:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls80_maxit100_3.log
.PHONY:t595
t595:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls80_maxit100_4.log
.PHONY:t596
t596:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls80_maxit100_5.log
.PHONY:t597
t597:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls80_maxit100_6.log
.PHONY:t598
t598:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls80_maxit100_7.log
.PHONY:t599
t599:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls80_maxit100_8.log
.PHONY:t600
t600:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls80_maxit100_9.log
.PHONY:t601
t601:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls100_maxit100_0.log
.PHONY:t602
t602:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls100_maxit100_1.log
.PHONY:t603
t603:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls100_maxit100_2.log
.PHONY:t604
t604:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls100_maxit100_3.log
.PHONY:t605
t605:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls100_maxit100_4.log
.PHONY:t606
t606:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls100_maxit100_5.log
.PHONY:t607
t607:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls100_maxit100_6.log
.PHONY:t608
t608:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls100_maxit100_7.log
.PHONY:t609
t609:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls100_maxit100_8.log
.PHONY:t610
t610:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls100_maxit100_9.log
.PHONY:t611
t611:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls150_maxit100_0.log
.PHONY:t612
t612:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls150_maxit100_1.log
.PHONY:t613
t613:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls150_maxit100_2.log
.PHONY:t614
t614:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls150_maxit100_3.log
.PHONY:t615
t615:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls150_maxit100_4.log
.PHONY:t616
t616:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls150_maxit100_5.log
.PHONY:t617
t617:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls150_maxit100_6.log
.PHONY:t618
t618:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls150_maxit100_7.log
.PHONY:t619
t619:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls150_maxit100_8.log
.PHONY:t620
t620:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls150_maxit100_9.log
.PHONY:t621
t621:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls200_maxit100_0.log
.PHONY:t622
t622:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls200_maxit100_1.log
.PHONY:t623
t623:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls200_maxit100_2.log
.PHONY:t624
t624:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls200_maxit100_3.log
.PHONY:t625
t625:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls200_maxit100_4.log
.PHONY:t626
t626:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls200_maxit100_5.log
.PHONY:t627
t627:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls200_maxit100_6.log
.PHONY:t628
t628:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls200_maxit100_7.log
.PHONY:t629
t629:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls200_maxit100_8.log
.PHONY:t630
t630:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.1_hls200_maxit100_9.log
.PHONY:t631
t631:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls2_maxit100_0.log
.PHONY:t632
t632:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls2_maxit100_1.log
.PHONY:t633
t633:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls2_maxit100_2.log
.PHONY:t634
t634:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls2_maxit100_3.log
.PHONY:t635
t635:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls2_maxit100_4.log
.PHONY:t636
t636:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls2_maxit100_5.log
.PHONY:t637
t637:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls2_maxit100_6.log
.PHONY:t638
t638:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls2_maxit100_7.log
.PHONY:t639
t639:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls2_maxit100_8.log
.PHONY:t640
t640:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls2_maxit100_9.log
.PHONY:t641
t641:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls5_maxit100_0.log
.PHONY:t642
t642:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls5_maxit100_1.log
.PHONY:t643
t643:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls5_maxit100_2.log
.PHONY:t644
t644:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls5_maxit100_3.log
.PHONY:t645
t645:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls5_maxit100_4.log
.PHONY:t646
t646:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls5_maxit100_5.log
.PHONY:t647
t647:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls5_maxit100_6.log
.PHONY:t648
t648:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls5_maxit100_7.log
.PHONY:t649
t649:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls5_maxit100_8.log
.PHONY:t650
t650:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls5_maxit100_9.log
.PHONY:t651
t651:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls10_maxit100_0.log
.PHONY:t652
t652:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls10_maxit100_1.log
.PHONY:t653
t653:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls10_maxit100_2.log
.PHONY:t654
t654:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls10_maxit100_3.log
.PHONY:t655
t655:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls10_maxit100_4.log
.PHONY:t656
t656:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls10_maxit100_5.log
.PHONY:t657
t657:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls10_maxit100_6.log
.PHONY:t658
t658:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls10_maxit100_7.log
.PHONY:t659
t659:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls10_maxit100_8.log
.PHONY:t660
t660:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls10_maxit100_9.log
.PHONY:t661
t661:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls30_maxit100_0.log
.PHONY:t662
t662:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls30_maxit100_1.log
.PHONY:t663
t663:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls30_maxit100_2.log
.PHONY:t664
t664:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls30_maxit100_3.log
.PHONY:t665
t665:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls30_maxit100_4.log
.PHONY:t666
t666:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls30_maxit100_5.log
.PHONY:t667
t667:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls30_maxit100_6.log
.PHONY:t668
t668:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls30_maxit100_7.log
.PHONY:t669
t669:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls30_maxit100_8.log
.PHONY:t670
t670:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls30_maxit100_9.log
.PHONY:t671
t671:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls50_maxit100_0.log
.PHONY:t672
t672:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls50_maxit100_1.log
.PHONY:t673
t673:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls50_maxit100_2.log
.PHONY:t674
t674:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls50_maxit100_3.log
.PHONY:t675
t675:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls50_maxit100_4.log
.PHONY:t676
t676:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls50_maxit100_5.log
.PHONY:t677
t677:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls50_maxit100_6.log
.PHONY:t678
t678:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls50_maxit100_7.log
.PHONY:t679
t679:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls50_maxit100_8.log
.PHONY:t680
t680:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls50_maxit100_9.log
.PHONY:t681
t681:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls80_maxit100_0.log
.PHONY:t682
t682:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls80_maxit100_1.log
.PHONY:t683
t683:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls80_maxit100_2.log
.PHONY:t684
t684:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls80_maxit100_3.log
.PHONY:t685
t685:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls80_maxit100_4.log
.PHONY:t686
t686:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls80_maxit100_5.log
.PHONY:t687
t687:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls80_maxit100_6.log
.PHONY:t688
t688:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls80_maxit100_7.log
.PHONY:t689
t689:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls80_maxit100_8.log
.PHONY:t690
t690:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls80_maxit100_9.log
.PHONY:t691
t691:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls100_maxit100_0.log
.PHONY:t692
t692:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls100_maxit100_1.log
.PHONY:t693
t693:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls100_maxit100_2.log
.PHONY:t694
t694:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls100_maxit100_3.log
.PHONY:t695
t695:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls100_maxit100_4.log
.PHONY:t696
t696:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls100_maxit100_5.log
.PHONY:t697
t697:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls100_maxit100_6.log
.PHONY:t698
t698:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls100_maxit100_7.log
.PHONY:t699
t699:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls100_maxit100_8.log
.PHONY:t700
t700:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls100_maxit100_9.log
.PHONY:t701
t701:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls150_maxit100_0.log
.PHONY:t702
t702:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls150_maxit100_1.log
.PHONY:t703
t703:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls150_maxit100_2.log
.PHONY:t704
t704:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls150_maxit100_3.log
.PHONY:t705
t705:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls150_maxit100_4.log
.PHONY:t706
t706:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls150_maxit100_5.log
.PHONY:t707
t707:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls150_maxit100_6.log
.PHONY:t708
t708:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls150_maxit100_7.log
.PHONY:t709
t709:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls150_maxit100_8.log
.PHONY:t710
t710:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls150_maxit100_9.log
.PHONY:t711
t711:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls200_maxit100_0.log
.PHONY:t712
t712:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls200_maxit100_1.log
.PHONY:t713
t713:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls200_maxit100_2.log
.PHONY:t714
t714:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls200_maxit100_3.log
.PHONY:t715
t715:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls200_maxit100_4.log
.PHONY:t716
t716:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls200_maxit100_5.log
.PHONY:t717
t717:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls200_maxit100_6.log
.PHONY:t718
t718:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls200_maxit100_7.log
.PHONY:t719
t719:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls200_maxit100_8.log
.PHONY:t720
t720:
	python deep_net.py \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_dense_init_he_uniform_vp0.2_hls200_maxit100_9.log
.PHONY:t721
t721:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls2_maxit100_0.log
.PHONY:t722
t722:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls2_maxit100_1.log
.PHONY:t723
t723:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls2_maxit100_2.log
.PHONY:t724
t724:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls2_maxit100_3.log
.PHONY:t725
t725:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls2_maxit100_4.log
.PHONY:t726
t726:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls2_maxit100_5.log
.PHONY:t727
t727:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls2_maxit100_6.log
.PHONY:t728
t728:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls2_maxit100_7.log
.PHONY:t729
t729:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls2_maxit100_8.log
.PHONY:t730
t730:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls2_maxit100_9.log
.PHONY:t731
t731:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls5_maxit100_0.log
.PHONY:t732
t732:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls5_maxit100_1.log
.PHONY:t733
t733:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls5_maxit100_2.log
.PHONY:t734
t734:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls5_maxit100_3.log
.PHONY:t735
t735:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls5_maxit100_4.log
.PHONY:t736
t736:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls5_maxit100_5.log
.PHONY:t737
t737:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls5_maxit100_6.log
.PHONY:t738
t738:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls5_maxit100_7.log
.PHONY:t739
t739:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls5_maxit100_8.log
.PHONY:t740
t740:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls5_maxit100_9.log
.PHONY:t741
t741:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls10_maxit100_0.log
.PHONY:t742
t742:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls10_maxit100_1.log
.PHONY:t743
t743:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls10_maxit100_2.log
.PHONY:t744
t744:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls10_maxit100_3.log
.PHONY:t745
t745:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls10_maxit100_4.log
.PHONY:t746
t746:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls10_maxit100_5.log
.PHONY:t747
t747:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls10_maxit100_6.log
.PHONY:t748
t748:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls10_maxit100_7.log
.PHONY:t749
t749:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls10_maxit100_8.log
.PHONY:t750
t750:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls10_maxit100_9.log
.PHONY:t751
t751:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls30_maxit100_0.log
.PHONY:t752
t752:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls30_maxit100_1.log
.PHONY:t753
t753:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls30_maxit100_2.log
.PHONY:t754
t754:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls30_maxit100_3.log
.PHONY:t755
t755:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls30_maxit100_4.log
.PHONY:t756
t756:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls30_maxit100_5.log
.PHONY:t757
t757:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls30_maxit100_6.log
.PHONY:t758
t758:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls30_maxit100_7.log
.PHONY:t759
t759:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls30_maxit100_8.log
.PHONY:t760
t760:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls30_maxit100_9.log
.PHONY:t761
t761:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls50_maxit100_0.log
.PHONY:t762
t762:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls50_maxit100_1.log
.PHONY:t763
t763:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls50_maxit100_2.log
.PHONY:t764
t764:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls50_maxit100_3.log
.PHONY:t765
t765:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls50_maxit100_4.log
.PHONY:t766
t766:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls50_maxit100_5.log
.PHONY:t767
t767:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls50_maxit100_6.log
.PHONY:t768
t768:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls50_maxit100_7.log
.PHONY:t769
t769:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls50_maxit100_8.log
.PHONY:t770
t770:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls50_maxit100_9.log
.PHONY:t771
t771:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls80_maxit100_0.log
.PHONY:t772
t772:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls80_maxit100_1.log
.PHONY:t773
t773:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls80_maxit100_2.log
.PHONY:t774
t774:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls80_maxit100_3.log
.PHONY:t775
t775:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls80_maxit100_4.log
.PHONY:t776
t776:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls80_maxit100_5.log
.PHONY:t777
t777:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls80_maxit100_6.log
.PHONY:t778
t778:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls80_maxit100_7.log
.PHONY:t779
t779:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls80_maxit100_8.log
.PHONY:t780
t780:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls80_maxit100_9.log
.PHONY:t781
t781:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls100_maxit100_0.log
.PHONY:t782
t782:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls100_maxit100_1.log
.PHONY:t783
t783:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls100_maxit100_2.log
.PHONY:t784
t784:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls100_maxit100_3.log
.PHONY:t785
t785:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls100_maxit100_4.log
.PHONY:t786
t786:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls100_maxit100_5.log
.PHONY:t787
t787:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls100_maxit100_6.log
.PHONY:t788
t788:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls100_maxit100_7.log
.PHONY:t789
t789:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls100_maxit100_8.log
.PHONY:t790
t790:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls100_maxit100_9.log
.PHONY:t791
t791:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls150_maxit100_0.log
.PHONY:t792
t792:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls150_maxit100_1.log
.PHONY:t793
t793:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls150_maxit100_2.log
.PHONY:t794
t794:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls150_maxit100_3.log
.PHONY:t795
t795:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls150_maxit100_4.log
.PHONY:t796
t796:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls150_maxit100_5.log
.PHONY:t797
t797:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls150_maxit100_6.log
.PHONY:t798
t798:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls150_maxit100_7.log
.PHONY:t799
t799:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls150_maxit100_8.log
.PHONY:t800
t800:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls150_maxit100_9.log
.PHONY:t801
t801:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls200_maxit100_0.log
.PHONY:t802
t802:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls200_maxit100_1.log
.PHONY:t803
t803:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls200_maxit100_2.log
.PHONY:t804
t804:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls200_maxit100_3.log
.PHONY:t805
t805:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls200_maxit100_4.log
.PHONY:t806
t806:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls200_maxit100_5.log
.PHONY:t807
t807:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls200_maxit100_6.log
.PHONY:t808
t808:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls200_maxit100_7.log
.PHONY:t809
t809:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls200_maxit100_8.log
.PHONY:t810
t810:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.1_hls200_maxit100_9.log
.PHONY:t811
t811:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls2_maxit100_0.log
.PHONY:t812
t812:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls2_maxit100_1.log
.PHONY:t813
t813:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls2_maxit100_2.log
.PHONY:t814
t814:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls2_maxit100_3.log
.PHONY:t815
t815:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls2_maxit100_4.log
.PHONY:t816
t816:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls2_maxit100_5.log
.PHONY:t817
t817:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls2_maxit100_6.log
.PHONY:t818
t818:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls2_maxit100_7.log
.PHONY:t819
t819:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls2_maxit100_8.log
.PHONY:t820
t820:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls2_maxit100_9.log
.PHONY:t821
t821:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls5_maxit100_0.log
.PHONY:t822
t822:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls5_maxit100_1.log
.PHONY:t823
t823:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls5_maxit100_2.log
.PHONY:t824
t824:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls5_maxit100_3.log
.PHONY:t825
t825:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls5_maxit100_4.log
.PHONY:t826
t826:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls5_maxit100_5.log
.PHONY:t827
t827:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls5_maxit100_6.log
.PHONY:t828
t828:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls5_maxit100_7.log
.PHONY:t829
t829:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls5_maxit100_8.log
.PHONY:t830
t830:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls5_maxit100_9.log
.PHONY:t831
t831:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls10_maxit100_0.log
.PHONY:t832
t832:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls10_maxit100_1.log
.PHONY:t833
t833:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls10_maxit100_2.log
.PHONY:t834
t834:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls10_maxit100_3.log
.PHONY:t835
t835:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls10_maxit100_4.log
.PHONY:t836
t836:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls10_maxit100_5.log
.PHONY:t837
t837:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls10_maxit100_6.log
.PHONY:t838
t838:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls10_maxit100_7.log
.PHONY:t839
t839:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls10_maxit100_8.log
.PHONY:t840
t840:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls10_maxit100_9.log
.PHONY:t841
t841:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls30_maxit100_0.log
.PHONY:t842
t842:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls30_maxit100_1.log
.PHONY:t843
t843:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls30_maxit100_2.log
.PHONY:t844
t844:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls30_maxit100_3.log
.PHONY:t845
t845:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls30_maxit100_4.log
.PHONY:t846
t846:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls30_maxit100_5.log
.PHONY:t847
t847:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls30_maxit100_6.log
.PHONY:t848
t848:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls30_maxit100_7.log
.PHONY:t849
t849:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls30_maxit100_8.log
.PHONY:t850
t850:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls30_maxit100_9.log
.PHONY:t851
t851:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls50_maxit100_0.log
.PHONY:t852
t852:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls50_maxit100_1.log
.PHONY:t853
t853:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls50_maxit100_2.log
.PHONY:t854
t854:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls50_maxit100_3.log
.PHONY:t855
t855:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls50_maxit100_4.log
.PHONY:t856
t856:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls50_maxit100_5.log
.PHONY:t857
t857:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls50_maxit100_6.log
.PHONY:t858
t858:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls50_maxit100_7.log
.PHONY:t859
t859:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls50_maxit100_8.log
.PHONY:t860
t860:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls50_maxit100_9.log
.PHONY:t861
t861:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls80_maxit100_0.log
.PHONY:t862
t862:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls80_maxit100_1.log
.PHONY:t863
t863:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls80_maxit100_2.log
.PHONY:t864
t864:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls80_maxit100_3.log
.PHONY:t865
t865:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls80_maxit100_4.log
.PHONY:t866
t866:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls80_maxit100_5.log
.PHONY:t867
t867:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls80_maxit100_6.log
.PHONY:t868
t868:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls80_maxit100_7.log
.PHONY:t869
t869:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls80_maxit100_8.log
.PHONY:t870
t870:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls80_maxit100_9.log
.PHONY:t871
t871:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls100_maxit100_0.log
.PHONY:t872
t872:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls100_maxit100_1.log
.PHONY:t873
t873:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls100_maxit100_2.log
.PHONY:t874
t874:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls100_maxit100_3.log
.PHONY:t875
t875:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls100_maxit100_4.log
.PHONY:t876
t876:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls100_maxit100_5.log
.PHONY:t877
t877:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls100_maxit100_6.log
.PHONY:t878
t878:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls100_maxit100_7.log
.PHONY:t879
t879:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls100_maxit100_8.log
.PHONY:t880
t880:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls100_maxit100_9.log
.PHONY:t881
t881:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls150_maxit100_0.log
.PHONY:t882
t882:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls150_maxit100_1.log
.PHONY:t883
t883:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls150_maxit100_2.log
.PHONY:t884
t884:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls150_maxit100_3.log
.PHONY:t885
t885:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls150_maxit100_4.log
.PHONY:t886
t886:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls150_maxit100_5.log
.PHONY:t887
t887:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls150_maxit100_6.log
.PHONY:t888
t888:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls150_maxit100_7.log
.PHONY:t889
t889:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls150_maxit100_8.log
.PHONY:t890
t890:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls150_maxit100_9.log
.PHONY:t891
t891:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls200_maxit100_0.log
.PHONY:t892
t892:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls200_maxit100_1.log
.PHONY:t893
t893:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls200_maxit100_2.log
.PHONY:t894
t894:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls200_maxit100_3.log
.PHONY:t895
t895:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls200_maxit100_4.log
.PHONY:t896
t896:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls200_maxit100_5.log
.PHONY:t897
t897:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls200_maxit100_6.log
.PHONY:t898
t898:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls200_maxit100_7.log
.PHONY:t899
t899:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls200_maxit100_8.log
.PHONY:t900
t900:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_uniform_vp0.2_hls200_maxit100_9.log
.PHONY:t901
t901:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls2_maxit100_0.log
.PHONY:t902
t902:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls2_maxit100_1.log
.PHONY:t903
t903:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls2_maxit100_2.log
.PHONY:t904
t904:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls2_maxit100_3.log
.PHONY:t905
t905:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls2_maxit100_4.log
.PHONY:t906
t906:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls2_maxit100_5.log
.PHONY:t907
t907:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls2_maxit100_6.log
.PHONY:t908
t908:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls2_maxit100_7.log
.PHONY:t909
t909:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls2_maxit100_8.log
.PHONY:t910
t910:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls2_maxit100_9.log
.PHONY:t911
t911:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls5_maxit100_0.log
.PHONY:t912
t912:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls5_maxit100_1.log
.PHONY:t913
t913:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls5_maxit100_2.log
.PHONY:t914
t914:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls5_maxit100_3.log
.PHONY:t915
t915:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls5_maxit100_4.log
.PHONY:t916
t916:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls5_maxit100_5.log
.PHONY:t917
t917:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls5_maxit100_6.log
.PHONY:t918
t918:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls5_maxit100_7.log
.PHONY:t919
t919:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls5_maxit100_8.log
.PHONY:t920
t920:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls5_maxit100_9.log
.PHONY:t921
t921:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls10_maxit100_0.log
.PHONY:t922
t922:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls10_maxit100_1.log
.PHONY:t923
t923:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls10_maxit100_2.log
.PHONY:t924
t924:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls10_maxit100_3.log
.PHONY:t925
t925:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls10_maxit100_4.log
.PHONY:t926
t926:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls10_maxit100_5.log
.PHONY:t927
t927:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls10_maxit100_6.log
.PHONY:t928
t928:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls10_maxit100_7.log
.PHONY:t929
t929:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls10_maxit100_8.log
.PHONY:t930
t930:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls10_maxit100_9.log
.PHONY:t931
t931:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls30_maxit100_0.log
.PHONY:t932
t932:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls30_maxit100_1.log
.PHONY:t933
t933:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls30_maxit100_2.log
.PHONY:t934
t934:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls30_maxit100_3.log
.PHONY:t935
t935:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls30_maxit100_4.log
.PHONY:t936
t936:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls30_maxit100_5.log
.PHONY:t937
t937:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls30_maxit100_6.log
.PHONY:t938
t938:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls30_maxit100_7.log
.PHONY:t939
t939:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls30_maxit100_8.log
.PHONY:t940
t940:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls30_maxit100_9.log
.PHONY:t941
t941:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls50_maxit100_0.log
.PHONY:t942
t942:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls50_maxit100_1.log
.PHONY:t943
t943:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls50_maxit100_2.log
.PHONY:t944
t944:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls50_maxit100_3.log
.PHONY:t945
t945:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls50_maxit100_4.log
.PHONY:t946
t946:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls50_maxit100_5.log
.PHONY:t947
t947:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls50_maxit100_6.log
.PHONY:t948
t948:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls50_maxit100_7.log
.PHONY:t949
t949:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls50_maxit100_8.log
.PHONY:t950
t950:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls50_maxit100_9.log
.PHONY:t951
t951:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls80_maxit100_0.log
.PHONY:t952
t952:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls80_maxit100_1.log
.PHONY:t953
t953:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls80_maxit100_2.log
.PHONY:t954
t954:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls80_maxit100_3.log
.PHONY:t955
t955:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls80_maxit100_4.log
.PHONY:t956
t956:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls80_maxit100_5.log
.PHONY:t957
t957:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls80_maxit100_6.log
.PHONY:t958
t958:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls80_maxit100_7.log
.PHONY:t959
t959:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls80_maxit100_8.log
.PHONY:t960
t960:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls80_maxit100_9.log
.PHONY:t961
t961:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls100_maxit100_0.log
.PHONY:t962
t962:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls100_maxit100_1.log
.PHONY:t963
t963:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls100_maxit100_2.log
.PHONY:t964
t964:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls100_maxit100_3.log
.PHONY:t965
t965:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls100_maxit100_4.log
.PHONY:t966
t966:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls100_maxit100_5.log
.PHONY:t967
t967:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls100_maxit100_6.log
.PHONY:t968
t968:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls100_maxit100_7.log
.PHONY:t969
t969:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls100_maxit100_8.log
.PHONY:t970
t970:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls100_maxit100_9.log
.PHONY:t971
t971:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls150_maxit100_0.log
.PHONY:t972
t972:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls150_maxit100_1.log
.PHONY:t973
t973:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls150_maxit100_2.log
.PHONY:t974
t974:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls150_maxit100_3.log
.PHONY:t975
t975:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls150_maxit100_4.log
.PHONY:t976
t976:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls150_maxit100_5.log
.PHONY:t977
t977:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls150_maxit100_6.log
.PHONY:t978
t978:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls150_maxit100_7.log
.PHONY:t979
t979:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls150_maxit100_8.log
.PHONY:t980
t980:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls150_maxit100_9.log
.PHONY:t981
t981:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls200_maxit100_0.log
.PHONY:t982
t982:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls200_maxit100_1.log
.PHONY:t983
t983:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls200_maxit100_2.log
.PHONY:t984
t984:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls200_maxit100_3.log
.PHONY:t985
t985:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls200_maxit100_4.log
.PHONY:t986
t986:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls200_maxit100_5.log
.PHONY:t987
t987:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls200_maxit100_6.log
.PHONY:t988
t988:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls200_maxit100_7.log
.PHONY:t989
t989:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls200_maxit100_8.log
.PHONY:t990
t990:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.1_hls200_maxit100_9.log
.PHONY:t991
t991:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls2_maxit100_0.log
.PHONY:t992
t992:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls2_maxit100_1.log
.PHONY:t993
t993:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls2_maxit100_2.log
.PHONY:t994
t994:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls2_maxit100_3.log
.PHONY:t995
t995:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls2_maxit100_4.log
.PHONY:t996
t996:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls2_maxit100_5.log
.PHONY:t997
t997:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls2_maxit100_6.log
.PHONY:t998
t998:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls2_maxit100_7.log
.PHONY:t999
t999:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls2_maxit100_8.log
.PHONY:t1000
t1000:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls2_maxit100_9.log
.PHONY:t1001
t1001:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls5_maxit100_0.log
.PHONY:t1002
t1002:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls5_maxit100_1.log
.PHONY:t1003
t1003:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls5_maxit100_2.log
.PHONY:t1004
t1004:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls5_maxit100_3.log
.PHONY:t1005
t1005:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls5_maxit100_4.log
.PHONY:t1006
t1006:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls5_maxit100_5.log
.PHONY:t1007
t1007:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls5_maxit100_6.log
.PHONY:t1008
t1008:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls5_maxit100_7.log
.PHONY:t1009
t1009:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls5_maxit100_8.log
.PHONY:t1010
t1010:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls5_maxit100_9.log
.PHONY:t1011
t1011:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls10_maxit100_0.log
.PHONY:t1012
t1012:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls10_maxit100_1.log
.PHONY:t1013
t1013:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls10_maxit100_2.log
.PHONY:t1014
t1014:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls10_maxit100_3.log
.PHONY:t1015
t1015:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls10_maxit100_4.log
.PHONY:t1016
t1016:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls10_maxit100_5.log
.PHONY:t1017
t1017:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls10_maxit100_6.log
.PHONY:t1018
t1018:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls10_maxit100_7.log
.PHONY:t1019
t1019:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls10_maxit100_8.log
.PHONY:t1020
t1020:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls10_maxit100_9.log
.PHONY:t1021
t1021:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls30_maxit100_0.log
.PHONY:t1022
t1022:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls30_maxit100_1.log
.PHONY:t1023
t1023:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls30_maxit100_2.log
.PHONY:t1024
t1024:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls30_maxit100_3.log
.PHONY:t1025
t1025:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls30_maxit100_4.log
.PHONY:t1026
t1026:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls30_maxit100_5.log
.PHONY:t1027
t1027:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls30_maxit100_6.log
.PHONY:t1028
t1028:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls30_maxit100_7.log
.PHONY:t1029
t1029:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls30_maxit100_8.log
.PHONY:t1030
t1030:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls30_maxit100_9.log
.PHONY:t1031
t1031:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls50_maxit100_0.log
.PHONY:t1032
t1032:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls50_maxit100_1.log
.PHONY:t1033
t1033:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls50_maxit100_2.log
.PHONY:t1034
t1034:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls50_maxit100_3.log
.PHONY:t1035
t1035:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls50_maxit100_4.log
.PHONY:t1036
t1036:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls50_maxit100_5.log
.PHONY:t1037
t1037:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls50_maxit100_6.log
.PHONY:t1038
t1038:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls50_maxit100_7.log
.PHONY:t1039
t1039:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls50_maxit100_8.log
.PHONY:t1040
t1040:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls50_maxit100_9.log
.PHONY:t1041
t1041:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls80_maxit100_0.log
.PHONY:t1042
t1042:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls80_maxit100_1.log
.PHONY:t1043
t1043:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls80_maxit100_2.log
.PHONY:t1044
t1044:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls80_maxit100_3.log
.PHONY:t1045
t1045:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls80_maxit100_4.log
.PHONY:t1046
t1046:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls80_maxit100_5.log
.PHONY:t1047
t1047:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls80_maxit100_6.log
.PHONY:t1048
t1048:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls80_maxit100_7.log
.PHONY:t1049
t1049:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls80_maxit100_8.log
.PHONY:t1050
t1050:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls80_maxit100_9.log
.PHONY:t1051
t1051:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls100_maxit100_0.log
.PHONY:t1052
t1052:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls100_maxit100_1.log
.PHONY:t1053
t1053:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls100_maxit100_2.log
.PHONY:t1054
t1054:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls100_maxit100_3.log
.PHONY:t1055
t1055:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls100_maxit100_4.log
.PHONY:t1056
t1056:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls100_maxit100_5.log
.PHONY:t1057
t1057:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls100_maxit100_6.log
.PHONY:t1058
t1058:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls100_maxit100_7.log
.PHONY:t1059
t1059:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls100_maxit100_8.log
.PHONY:t1060
t1060:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls100_maxit100_9.log
.PHONY:t1061
t1061:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls150_maxit100_0.log
.PHONY:t1062
t1062:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls150_maxit100_1.log
.PHONY:t1063
t1063:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls150_maxit100_2.log
.PHONY:t1064
t1064:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls150_maxit100_3.log
.PHONY:t1065
t1065:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls150_maxit100_4.log
.PHONY:t1066
t1066:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls150_maxit100_5.log
.PHONY:t1067
t1067:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls150_maxit100_6.log
.PHONY:t1068
t1068:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls150_maxit100_7.log
.PHONY:t1069
t1069:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls150_maxit100_8.log
.PHONY:t1070
t1070:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls150_maxit100_9.log
.PHONY:t1071
t1071:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls200_maxit100_0.log
.PHONY:t1072
t1072:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls200_maxit100_1.log
.PHONY:t1073
t1073:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls200_maxit100_2.log
.PHONY:t1074
t1074:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls200_maxit100_3.log
.PHONY:t1075
t1075:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls200_maxit100_4.log
.PHONY:t1076
t1076:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls200_maxit100_5.log
.PHONY:t1077
t1077:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls200_maxit100_6.log
.PHONY:t1078
t1078:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls200_maxit100_7.log
.PHONY:t1079
t1079:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls200_maxit100_8.log
.PHONY:t1080
t1080:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init glorot_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_glorot_normal_vp0.2_hls200_maxit100_9.log
.PHONY:t1081
t1081:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls2_maxit100_0.log
.PHONY:t1082
t1082:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls2_maxit100_1.log
.PHONY:t1083
t1083:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls2_maxit100_2.log
.PHONY:t1084
t1084:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls2_maxit100_3.log
.PHONY:t1085
t1085:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls2_maxit100_4.log
.PHONY:t1086
t1086:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls2_maxit100_5.log
.PHONY:t1087
t1087:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls2_maxit100_6.log
.PHONY:t1088
t1088:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls2_maxit100_7.log
.PHONY:t1089
t1089:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls2_maxit100_8.log
.PHONY:t1090
t1090:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls2_maxit100_9.log
.PHONY:t1091
t1091:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls5_maxit100_0.log
.PHONY:t1092
t1092:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls5_maxit100_1.log
.PHONY:t1093
t1093:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls5_maxit100_2.log
.PHONY:t1094
t1094:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls5_maxit100_3.log
.PHONY:t1095
t1095:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls5_maxit100_4.log
.PHONY:t1096
t1096:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls5_maxit100_5.log
.PHONY:t1097
t1097:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls5_maxit100_6.log
.PHONY:t1098
t1098:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls5_maxit100_7.log
.PHONY:t1099
t1099:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls5_maxit100_8.log
.PHONY:t1100
t1100:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls5_maxit100_9.log
.PHONY:t1101
t1101:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls10_maxit100_0.log
.PHONY:t1102
t1102:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls10_maxit100_1.log
.PHONY:t1103
t1103:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls10_maxit100_2.log
.PHONY:t1104
t1104:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls10_maxit100_3.log
.PHONY:t1105
t1105:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls10_maxit100_4.log
.PHONY:t1106
t1106:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls10_maxit100_5.log
.PHONY:t1107
t1107:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls10_maxit100_6.log
.PHONY:t1108
t1108:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls10_maxit100_7.log
.PHONY:t1109
t1109:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls10_maxit100_8.log
.PHONY:t1110
t1110:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls10_maxit100_9.log
.PHONY:t1111
t1111:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls30_maxit100_0.log
.PHONY:t1112
t1112:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls30_maxit100_1.log
.PHONY:t1113
t1113:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls30_maxit100_2.log
.PHONY:t1114
t1114:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls30_maxit100_3.log
.PHONY:t1115
t1115:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls30_maxit100_4.log
.PHONY:t1116
t1116:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls30_maxit100_5.log
.PHONY:t1117
t1117:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls30_maxit100_6.log
.PHONY:t1118
t1118:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls30_maxit100_7.log
.PHONY:t1119
t1119:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls30_maxit100_8.log
.PHONY:t1120
t1120:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls30_maxit100_9.log
.PHONY:t1121
t1121:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls50_maxit100_0.log
.PHONY:t1122
t1122:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls50_maxit100_1.log
.PHONY:t1123
t1123:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls50_maxit100_2.log
.PHONY:t1124
t1124:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls50_maxit100_3.log
.PHONY:t1125
t1125:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls50_maxit100_4.log
.PHONY:t1126
t1126:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls50_maxit100_5.log
.PHONY:t1127
t1127:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls50_maxit100_6.log
.PHONY:t1128
t1128:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls50_maxit100_7.log
.PHONY:t1129
t1129:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls50_maxit100_8.log
.PHONY:t1130
t1130:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls50_maxit100_9.log
.PHONY:t1131
t1131:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls80_maxit100_0.log
.PHONY:t1132
t1132:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls80_maxit100_1.log
.PHONY:t1133
t1133:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls80_maxit100_2.log
.PHONY:t1134
t1134:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls80_maxit100_3.log
.PHONY:t1135
t1135:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls80_maxit100_4.log
.PHONY:t1136
t1136:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls80_maxit100_5.log
.PHONY:t1137
t1137:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls80_maxit100_6.log
.PHONY:t1138
t1138:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls80_maxit100_7.log
.PHONY:t1139
t1139:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls80_maxit100_8.log
.PHONY:t1140
t1140:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls80_maxit100_9.log
.PHONY:t1141
t1141:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls100_maxit100_0.log
.PHONY:t1142
t1142:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls100_maxit100_1.log
.PHONY:t1143
t1143:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls100_maxit100_2.log
.PHONY:t1144
t1144:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls100_maxit100_3.log
.PHONY:t1145
t1145:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls100_maxit100_4.log
.PHONY:t1146
t1146:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls100_maxit100_5.log
.PHONY:t1147
t1147:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls100_maxit100_6.log
.PHONY:t1148
t1148:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls100_maxit100_7.log
.PHONY:t1149
t1149:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls100_maxit100_8.log
.PHONY:t1150
t1150:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls100_maxit100_9.log
.PHONY:t1151
t1151:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls150_maxit100_0.log
.PHONY:t1152
t1152:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls150_maxit100_1.log
.PHONY:t1153
t1153:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls150_maxit100_2.log
.PHONY:t1154
t1154:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls150_maxit100_3.log
.PHONY:t1155
t1155:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls150_maxit100_4.log
.PHONY:t1156
t1156:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls150_maxit100_5.log
.PHONY:t1157
t1157:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls150_maxit100_6.log
.PHONY:t1158
t1158:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls150_maxit100_7.log
.PHONY:t1159
t1159:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls150_maxit100_8.log
.PHONY:t1160
t1160:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls150_maxit100_9.log
.PHONY:t1161
t1161:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls200_maxit100_0.log
.PHONY:t1162
t1162:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls200_maxit100_1.log
.PHONY:t1163
t1163:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls200_maxit100_2.log
.PHONY:t1164
t1164:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls200_maxit100_3.log
.PHONY:t1165
t1165:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls200_maxit100_4.log
.PHONY:t1166
t1166:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls200_maxit100_5.log
.PHONY:t1167
t1167:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls200_maxit100_6.log
.PHONY:t1168
t1168:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls200_maxit100_7.log
.PHONY:t1169
t1169:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls200_maxit100_8.log
.PHONY:t1170
t1170:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.1_hls200_maxit100_9.log
.PHONY:t1171
t1171:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls2_maxit100_0.log
.PHONY:t1172
t1172:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls2_maxit100_1.log
.PHONY:t1173
t1173:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls2_maxit100_2.log
.PHONY:t1174
t1174:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls2_maxit100_3.log
.PHONY:t1175
t1175:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls2_maxit100_4.log
.PHONY:t1176
t1176:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls2_maxit100_5.log
.PHONY:t1177
t1177:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls2_maxit100_6.log
.PHONY:t1178
t1178:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls2_maxit100_7.log
.PHONY:t1179
t1179:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls2_maxit100_8.log
.PHONY:t1180
t1180:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls2_maxit100_9.log
.PHONY:t1181
t1181:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls5_maxit100_0.log
.PHONY:t1182
t1182:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls5_maxit100_1.log
.PHONY:t1183
t1183:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls5_maxit100_2.log
.PHONY:t1184
t1184:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls5_maxit100_3.log
.PHONY:t1185
t1185:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls5_maxit100_4.log
.PHONY:t1186
t1186:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls5_maxit100_5.log
.PHONY:t1187
t1187:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls5_maxit100_6.log
.PHONY:t1188
t1188:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls5_maxit100_7.log
.PHONY:t1189
t1189:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls5_maxit100_8.log
.PHONY:t1190
t1190:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls5_maxit100_9.log
.PHONY:t1191
t1191:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls10_maxit100_0.log
.PHONY:t1192
t1192:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls10_maxit100_1.log
.PHONY:t1193
t1193:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls10_maxit100_2.log
.PHONY:t1194
t1194:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls10_maxit100_3.log
.PHONY:t1195
t1195:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls10_maxit100_4.log
.PHONY:t1196
t1196:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls10_maxit100_5.log
.PHONY:t1197
t1197:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls10_maxit100_6.log
.PHONY:t1198
t1198:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls10_maxit100_7.log
.PHONY:t1199
t1199:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls10_maxit100_8.log
.PHONY:t1200
t1200:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls10_maxit100_9.log
.PHONY:t1201
t1201:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls30_maxit100_0.log
.PHONY:t1202
t1202:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls30_maxit100_1.log
.PHONY:t1203
t1203:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls30_maxit100_2.log
.PHONY:t1204
t1204:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls30_maxit100_3.log
.PHONY:t1205
t1205:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls30_maxit100_4.log
.PHONY:t1206
t1206:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls30_maxit100_5.log
.PHONY:t1207
t1207:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls30_maxit100_6.log
.PHONY:t1208
t1208:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls30_maxit100_7.log
.PHONY:t1209
t1209:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls30_maxit100_8.log
.PHONY:t1210
t1210:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls30_maxit100_9.log
.PHONY:t1211
t1211:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls50_maxit100_0.log
.PHONY:t1212
t1212:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls50_maxit100_1.log
.PHONY:t1213
t1213:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls50_maxit100_2.log
.PHONY:t1214
t1214:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls50_maxit100_3.log
.PHONY:t1215
t1215:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls50_maxit100_4.log
.PHONY:t1216
t1216:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls50_maxit100_5.log
.PHONY:t1217
t1217:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls50_maxit100_6.log
.PHONY:t1218
t1218:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls50_maxit100_7.log
.PHONY:t1219
t1219:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls50_maxit100_8.log
.PHONY:t1220
t1220:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls50_maxit100_9.log
.PHONY:t1221
t1221:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls80_maxit100_0.log
.PHONY:t1222
t1222:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls80_maxit100_1.log
.PHONY:t1223
t1223:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls80_maxit100_2.log
.PHONY:t1224
t1224:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls80_maxit100_3.log
.PHONY:t1225
t1225:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls80_maxit100_4.log
.PHONY:t1226
t1226:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls80_maxit100_5.log
.PHONY:t1227
t1227:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls80_maxit100_6.log
.PHONY:t1228
t1228:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls80_maxit100_7.log
.PHONY:t1229
t1229:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls80_maxit100_8.log
.PHONY:t1230
t1230:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls80_maxit100_9.log
.PHONY:t1231
t1231:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls100_maxit100_0.log
.PHONY:t1232
t1232:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls100_maxit100_1.log
.PHONY:t1233
t1233:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls100_maxit100_2.log
.PHONY:t1234
t1234:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls100_maxit100_3.log
.PHONY:t1235
t1235:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls100_maxit100_4.log
.PHONY:t1236
t1236:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls100_maxit100_5.log
.PHONY:t1237
t1237:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls100_maxit100_6.log
.PHONY:t1238
t1238:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls100_maxit100_7.log
.PHONY:t1239
t1239:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls100_maxit100_8.log
.PHONY:t1240
t1240:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls100_maxit100_9.log
.PHONY:t1241
t1241:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls150_maxit100_0.log
.PHONY:t1242
t1242:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls150_maxit100_1.log
.PHONY:t1243
t1243:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls150_maxit100_2.log
.PHONY:t1244
t1244:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls150_maxit100_3.log
.PHONY:t1245
t1245:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls150_maxit100_4.log
.PHONY:t1246
t1246:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls150_maxit100_5.log
.PHONY:t1247
t1247:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls150_maxit100_6.log
.PHONY:t1248
t1248:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls150_maxit100_7.log
.PHONY:t1249
t1249:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls150_maxit100_8.log
.PHONY:t1250
t1250:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls150_maxit100_9.log
.PHONY:t1251
t1251:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls200_maxit100_0.log
.PHONY:t1252
t1252:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls200_maxit100_1.log
.PHONY:t1253
t1253:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls200_maxit100_2.log
.PHONY:t1254
t1254:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls200_maxit100_3.log
.PHONY:t1255
t1255:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls200_maxit100_4.log
.PHONY:t1256
t1256:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls200_maxit100_5.log
.PHONY:t1257
t1257:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls200_maxit100_6.log
.PHONY:t1258
t1258:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls200_maxit100_7.log
.PHONY:t1259
t1259:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls200_maxit100_8.log
.PHONY:t1260
t1260:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_normal \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_normal_vp0.2_hls200_maxit100_9.log
.PHONY:t1261
t1261:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls2_maxit100_0.log
.PHONY:t1262
t1262:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls2_maxit100_1.log
.PHONY:t1263
t1263:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls2_maxit100_2.log
.PHONY:t1264
t1264:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls2_maxit100_3.log
.PHONY:t1265
t1265:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls2_maxit100_4.log
.PHONY:t1266
t1266:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls2_maxit100_5.log
.PHONY:t1267
t1267:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls2_maxit100_6.log
.PHONY:t1268
t1268:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls2_maxit100_7.log
.PHONY:t1269
t1269:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls2_maxit100_8.log
.PHONY:t1270
t1270:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls2_maxit100_9.log
.PHONY:t1271
t1271:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls5_maxit100_0.log
.PHONY:t1272
t1272:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls5_maxit100_1.log
.PHONY:t1273
t1273:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls5_maxit100_2.log
.PHONY:t1274
t1274:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls5_maxit100_3.log
.PHONY:t1275
t1275:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls5_maxit100_4.log
.PHONY:t1276
t1276:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls5_maxit100_5.log
.PHONY:t1277
t1277:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls5_maxit100_6.log
.PHONY:t1278
t1278:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls5_maxit100_7.log
.PHONY:t1279
t1279:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls5_maxit100_8.log
.PHONY:t1280
t1280:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls5_maxit100_9.log
.PHONY:t1281
t1281:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls10_maxit100_0.log
.PHONY:t1282
t1282:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls10_maxit100_1.log
.PHONY:t1283
t1283:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls10_maxit100_2.log
.PHONY:t1284
t1284:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls10_maxit100_3.log
.PHONY:t1285
t1285:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls10_maxit100_4.log
.PHONY:t1286
t1286:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls10_maxit100_5.log
.PHONY:t1287
t1287:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls10_maxit100_6.log
.PHONY:t1288
t1288:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls10_maxit100_7.log
.PHONY:t1289
t1289:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls10_maxit100_8.log
.PHONY:t1290
t1290:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls10_maxit100_9.log
.PHONY:t1291
t1291:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls30_maxit100_0.log
.PHONY:t1292
t1292:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls30_maxit100_1.log
.PHONY:t1293
t1293:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls30_maxit100_2.log
.PHONY:t1294
t1294:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls30_maxit100_3.log
.PHONY:t1295
t1295:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls30_maxit100_4.log
.PHONY:t1296
t1296:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls30_maxit100_5.log
.PHONY:t1297
t1297:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls30_maxit100_6.log
.PHONY:t1298
t1298:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls30_maxit100_7.log
.PHONY:t1299
t1299:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls30_maxit100_8.log
.PHONY:t1300
t1300:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls30_maxit100_9.log
.PHONY:t1301
t1301:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls50_maxit100_0.log
.PHONY:t1302
t1302:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls50_maxit100_1.log
.PHONY:t1303
t1303:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls50_maxit100_2.log
.PHONY:t1304
t1304:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls50_maxit100_3.log
.PHONY:t1305
t1305:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls50_maxit100_4.log
.PHONY:t1306
t1306:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls50_maxit100_5.log
.PHONY:t1307
t1307:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls50_maxit100_6.log
.PHONY:t1308
t1308:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls50_maxit100_7.log
.PHONY:t1309
t1309:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls50_maxit100_8.log
.PHONY:t1310
t1310:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls50_maxit100_9.log
.PHONY:t1311
t1311:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls80_maxit100_0.log
.PHONY:t1312
t1312:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls80_maxit100_1.log
.PHONY:t1313
t1313:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls80_maxit100_2.log
.PHONY:t1314
t1314:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls80_maxit100_3.log
.PHONY:t1315
t1315:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls80_maxit100_4.log
.PHONY:t1316
t1316:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls80_maxit100_5.log
.PHONY:t1317
t1317:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls80_maxit100_6.log
.PHONY:t1318
t1318:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls80_maxit100_7.log
.PHONY:t1319
t1319:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls80_maxit100_8.log
.PHONY:t1320
t1320:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls80_maxit100_9.log
.PHONY:t1321
t1321:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls100_maxit100_0.log
.PHONY:t1322
t1322:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls100_maxit100_1.log
.PHONY:t1323
t1323:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls100_maxit100_2.log
.PHONY:t1324
t1324:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls100_maxit100_3.log
.PHONY:t1325
t1325:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls100_maxit100_4.log
.PHONY:t1326
t1326:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls100_maxit100_5.log
.PHONY:t1327
t1327:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls100_maxit100_6.log
.PHONY:t1328
t1328:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls100_maxit100_7.log
.PHONY:t1329
t1329:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls100_maxit100_8.log
.PHONY:t1330
t1330:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls100_maxit100_9.log
.PHONY:t1331
t1331:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls150_maxit100_0.log
.PHONY:t1332
t1332:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls150_maxit100_1.log
.PHONY:t1333
t1333:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls150_maxit100_2.log
.PHONY:t1334
t1334:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls150_maxit100_3.log
.PHONY:t1335
t1335:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls150_maxit100_4.log
.PHONY:t1336
t1336:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls150_maxit100_5.log
.PHONY:t1337
t1337:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls150_maxit100_6.log
.PHONY:t1338
t1338:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls150_maxit100_7.log
.PHONY:t1339
t1339:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls150_maxit100_8.log
.PHONY:t1340
t1340:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls150_maxit100_9.log
.PHONY:t1341
t1341:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls200_maxit100_0.log
.PHONY:t1342
t1342:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls200_maxit100_1.log
.PHONY:t1343
t1343:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls200_maxit100_2.log
.PHONY:t1344
t1344:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls200_maxit100_3.log
.PHONY:t1345
t1345:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls200_maxit100_4.log
.PHONY:t1346
t1346:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls200_maxit100_5.log
.PHONY:t1347
t1347:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls200_maxit100_6.log
.PHONY:t1348
t1348:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls200_maxit100_7.log
.PHONY:t1349
t1349:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls200_maxit100_8.log
.PHONY:t1350
t1350:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.1 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.1_hls200_maxit100_9.log
.PHONY:t1351
t1351:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls2_maxit100_0.log
.PHONY:t1352
t1352:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls2_maxit100_1.log
.PHONY:t1353
t1353:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls2_maxit100_2.log
.PHONY:t1354
t1354:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls2_maxit100_3.log
.PHONY:t1355
t1355:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls2_maxit100_4.log
.PHONY:t1356
t1356:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls2_maxit100_5.log
.PHONY:t1357
t1357:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls2_maxit100_6.log
.PHONY:t1358
t1358:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls2_maxit100_7.log
.PHONY:t1359
t1359:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls2_maxit100_8.log
.PHONY:t1360
t1360:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 2 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 2 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls2_maxit100_9.log
.PHONY:t1361
t1361:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls5_maxit100_0.log
.PHONY:t1362
t1362:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls5_maxit100_1.log
.PHONY:t1363
t1363:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls5_maxit100_2.log
.PHONY:t1364
t1364:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls5_maxit100_3.log
.PHONY:t1365
t1365:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls5_maxit100_4.log
.PHONY:t1366
t1366:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls5_maxit100_5.log
.PHONY:t1367
t1367:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls5_maxit100_6.log
.PHONY:t1368
t1368:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls5_maxit100_7.log
.PHONY:t1369
t1369:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls5_maxit100_8.log
.PHONY:t1370
t1370:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 5 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 5 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls5_maxit100_9.log
.PHONY:t1371
t1371:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls10_maxit100_0.log
.PHONY:t1372
t1372:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls10_maxit100_1.log
.PHONY:t1373
t1373:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls10_maxit100_2.log
.PHONY:t1374
t1374:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls10_maxit100_3.log
.PHONY:t1375
t1375:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls10_maxit100_4.log
.PHONY:t1376
t1376:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls10_maxit100_5.log
.PHONY:t1377
t1377:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls10_maxit100_6.log
.PHONY:t1378
t1378:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls10_maxit100_7.log
.PHONY:t1379
t1379:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls10_maxit100_8.log
.PHONY:t1380
t1380:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 10 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 10 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls10_maxit100_9.log
.PHONY:t1381
t1381:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls30_maxit100_0.log
.PHONY:t1382
t1382:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls30_maxit100_1.log
.PHONY:t1383
t1383:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls30_maxit100_2.log
.PHONY:t1384
t1384:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls30_maxit100_3.log
.PHONY:t1385
t1385:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls30_maxit100_4.log
.PHONY:t1386
t1386:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls30_maxit100_5.log
.PHONY:t1387
t1387:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls30_maxit100_6.log
.PHONY:t1388
t1388:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls30_maxit100_7.log
.PHONY:t1389
t1389:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls30_maxit100_8.log
.PHONY:t1390
t1390:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 30 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 30 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls30_maxit100_9.log
.PHONY:t1391
t1391:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls50_maxit100_0.log
.PHONY:t1392
t1392:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls50_maxit100_1.log
.PHONY:t1393
t1393:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls50_maxit100_2.log
.PHONY:t1394
t1394:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls50_maxit100_3.log
.PHONY:t1395
t1395:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls50_maxit100_4.log
.PHONY:t1396
t1396:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls50_maxit100_5.log
.PHONY:t1397
t1397:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls50_maxit100_6.log
.PHONY:t1398
t1398:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls50_maxit100_7.log
.PHONY:t1399
t1399:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls50_maxit100_8.log
.PHONY:t1400
t1400:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 50 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 50 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls50_maxit100_9.log
.PHONY:t1401
t1401:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls80_maxit100_0.log
.PHONY:t1402
t1402:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls80_maxit100_1.log
.PHONY:t1403
t1403:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls80_maxit100_2.log
.PHONY:t1404
t1404:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls80_maxit100_3.log
.PHONY:t1405
t1405:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls80_maxit100_4.log
.PHONY:t1406
t1406:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls80_maxit100_5.log
.PHONY:t1407
t1407:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls80_maxit100_6.log
.PHONY:t1408
t1408:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls80_maxit100_7.log
.PHONY:t1409
t1409:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls80_maxit100_8.log
.PHONY:t1410
t1410:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 80 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 80 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls80_maxit100_9.log
.PHONY:t1411
t1411:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls100_maxit100_0.log
.PHONY:t1412
t1412:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls100_maxit100_1.log
.PHONY:t1413
t1413:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls100_maxit100_2.log
.PHONY:t1414
t1414:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls100_maxit100_3.log
.PHONY:t1415
t1415:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls100_maxit100_4.log
.PHONY:t1416
t1416:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls100_maxit100_5.log
.PHONY:t1417
t1417:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls100_maxit100_6.log
.PHONY:t1418
t1418:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls100_maxit100_7.log
.PHONY:t1419
t1419:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls100_maxit100_8.log
.PHONY:t1420
t1420:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 100 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 100 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls100_maxit100_9.log
.PHONY:t1421
t1421:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls150_maxit100_0.log
.PHONY:t1422
t1422:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls150_maxit100_1.log
.PHONY:t1423
t1423:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls150_maxit100_2.log
.PHONY:t1424
t1424:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls150_maxit100_3.log
.PHONY:t1425
t1425:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls150_maxit100_4.log
.PHONY:t1426
t1426:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls150_maxit100_5.log
.PHONY:t1427
t1427:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls150_maxit100_6.log
.PHONY:t1428
t1428:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls150_maxit100_7.log
.PHONY:t1429
t1429:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls150_maxit100_8.log
.PHONY:t1430
t1430:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 150 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 150 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls150_maxit100_9.log
.PHONY:t1431
t1431:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls200_maxit100_0.log
.PHONY:t1432
t1432:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls200_maxit100_1.log
.PHONY:t1433
t1433:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls200_maxit100_2.log
.PHONY:t1434
t1434:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls200_maxit100_3.log
.PHONY:t1435
t1435:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls200_maxit100_4.log
.PHONY:t1436
t1436:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls200_maxit100_5.log
.PHONY:t1437
t1437:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls200_maxit100_6.log
.PHONY:t1438
t1438:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls200_maxit100_7.log
.PHONY:t1439
t1439:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls200_maxit100_8.log
.PHONY:t1440
t1440:
	python deep_net.py \
	-p ppi_tf_merge_cluster.txt \
	-merge 200 \
	-d important_file/TPM_mouse_7_8_10_PPITF_gene_9437.txt \
	-init he_uniform \
	-m 100 \
	-hls 200 \
	-stc 1 \
	-vp 0.2 \
	> log_deep_param/1layer_PPITF_init_he_uniform_vp0.2_hls200_maxit100_9.log
