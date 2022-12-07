import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

[0.07238201941937616, 0.0800935398379009, 0.08890424265641056, 0.10201232595598861, 3.2316958881404303]
t2m = [[0.0722093029687666, 0.0800599765746969, 0.08814243414018252, 0.1014979167343819, 3.2501960303685435], [0.07229515226838523, 0.08004019900850237, 0.08852833257350202, 0.10198208941784624, 3.251752939289563], [0.07238201941937616, 0.0800935398379009, 0.08890424265641056, 0.10201232595598861, 3.2316958881404303], [0.07263665723371995, 0.08035904435976728, 0.08906469484918739, 0.1025458826900345, 3.2269310356819467], [0.26520073758820967, 0.2663207660799157, 0.2644291222707866, 0.2673593182269841, 4.040020259439129]]
u10 = [[0.07220905352741072, 0.07775378690598762, 0.08347954417336477, 0.09076263350370813, 2.396611103782915], [0.07428738698873617, 0.08016264293189734, 0.08561618764923043, 0.09300113430578415, 2.3818825445763054], [0.07697698448414672, 0.08290545499488099, 0.08875801520396585, 0.0970862221738247, 2.3910398651475777], [0.08142540476297679, 0.08711862695747859, 0.09346292453677688, 0.10255276530164562, 2.3901029818678556], [0.3764110031601501, 0.3738983541318815, 0.3795340845437899, 0.3858765508214088, 4.027153139898222]]
v10 = [[0.072209004547498, 0.07904815762622716, 0.08593964525689818, 0.09491961908667054, 3.0416099190711976], [0.07286407560722469, 0.08010144530195897, 0.0872788974173265, 0.09607718883924288, 3.0539444925033883], [0.07406204691386387, 0.08108673912205108, 0.08850853192071392, 0.09830532874146553, 3.0320396279635493], [0.07624512328064605, 0.0839645757965029, 0.09156712360986291, 0.10291724500999058, 3.037093996511747], [0.31619392762037174, 0.31475630559741635, 0.31328867955567086, 0.3124570830023452, 4.032761159661698]]
t = [[0.07220813128331753, 0.07505847449886473, 0.08030619591168345, 0.08980761931775368, 1.7222918887660927], [0.07676110899917883, 0.07998217136076052, 0.08479389837649587, 0.09517661739293844, 1.7303510270706595], [0.08030846610869447, 0.08332172555160032, 0.08836342342299958, 0.09844963105574046, 1.7313192150364183], [0.0835013199444503, 0.08685569634074218, 0.09210758835893788, 0.102483169326227, 1.7292524919117966], [0.4228802060427731, 0.42367351500138845, 0.42381612544190395, 0.42888073447632463, 4.058897024638032]]
tcc = [[0.07221011389812378, 0.07987452751051073, 0.08801318289686556, 0.10170070316693554, 3.880199689734472], [0.07233366519723036, 0.07999831099216252, 0.08821563472690648, 0.10111718176774782, 3.8888550222736518], [0.0725501667912284, 0.08027569475545458, 0.08843710169400254, 0.10207627590184343, 3.892108341112529], [0.07303686244030522, 0.08063413470677316, 0.08889956531459338, 0.1025500566685853, 3.8675567094593832], [0.25249958028123803, 0.25026642647508074, 0.2477248637643579, 0.2441681728787618, 4.037529154019813]]
tp = [[0.07221022444738917, 0.0797339978997838, 0.08837135939361299, 0.10197527041173961, 3.638224976356715], [0.07228166942727075, 0.07968098944282695, 0.08847817952094013, 0.10201462556238043, 3.663897924553858], [0.07247019648449878, 0.08024300851234017, 0.08850968973073241, 0.10190408540098635, 3.661444519970515], [0.07271822700149393, 0.08044336396333289, 0.0889157191123048, 0.10251743701631076, 3.6247202471511004], [0.25206399692656245, 0.2498923100633164, 0.2525695261685816, 0.25180741987408023, 4.037137409432294]]

t2m_heat = sns.heatmap(t2m)
u10_heat = sns.heatmap(u10)
v10_heat = sns.heatmap(v10)
t_heat = sns.heatmap(t)
tcc_heat = sns.heatmap(tcc)
tp_heat = sns.heatmap(tp)

plt.show()
figure1 = t2m_heat.get_figure()
figure2 = u10_heat.get_figure()
figure3 = v10_heat.get_figure()
figure4 = t_heat.get_figure()
figure5 = tcc_heat.get_figure()
figure6 = tp_heat.get_figure()
figure1.savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/z_analysis_t2m_heatmap')
figure1.savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/z_analysis_u10_heatmap')
figure1.savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/z_analysis_v10_heatmap')
figure1.savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/z_analysis_t_heatmap')
figure1.savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/z_analysis_tcc_heatmap')
figure1.savefig('/home/ge75tis/Desktop/Thesis-project-CNN/Graphs/z_analysis_tp_heatmap')
