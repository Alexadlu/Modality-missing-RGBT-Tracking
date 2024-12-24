import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings

class LasHeR_trainingSet(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, data_fraction=None, attr=None):
        self.root = env_settings().lasher_dir if root is None else root
        super().__init__('LasHeR_trainingSet', root, image_loader)

        # video_name for each sequence
        self.sequence_list = ['2ndblkboy1_quezhen', '2ndrunningboy', 'abreastinnerboy', 'ajiandan_blkdog', 'ajiandan_boyleft2right', 'ajiandan_catwhite', 'basketballatright', 'basketballup', 'bike2', 'bikeafterwhitecar', 'bikeboy128', 'blkbagontheleftgirl', 'blkboy', 'blkboycoming', 'blkboygoleft', 'blkboylefttheredbagboy', 'blkboywillstand', 'blkboywithbluebag', 'blkboywithglasses', 'blkboywithwhitebackpack', 'blkgirlfat_quezhen', 'blkgirlfromcolumn1_quezhen', 'blkteacher`shead', 'bluegirlcoming', 'bluegirlriding', 'bowblkboy1-quezhen', 'boy', 'boy1227', 'boyaftertworunboys', 'boyatbluegirlleft', 'boybackpack', 'boybehindtrees', 'boybehindtrees2', 'boycoming', 'boyindarkwithgirl', 'boyinsnowfield2', 'boyinsnowfield4', 'boyinsnowfield_inf_white', 'boyleft', 'boyplayingphone', 'boyputtinghandup', 'boyrightrubbish', 'boyrightthelightbrown', 'boyrunning', 'boytakingcamera', 'boytoleft_inf_white', 'boyunderthecolumn', 'boywalkinginsnow', 'boywalkinginsnow3', 'car', 'carleaves', "comingboy'shead", 'easy_rightblkboywithgirl', 'easy_runninggirls', 'easy_whiterignt2left', 'farmanrightwhitesmallhouse', 'firstleftrunning', 'fogboyscoming1_quezhen_inf_heiying', 'girlatleft', 'girlruns2right', 'girltakingplate', 'girlwithredhat', 'greenboy', 'greenboyafterwhite', 'greyboysit1_quezhen', 'hatboy`shead', 'lastblkboy1_quezhen', 'left2ndboy', 'left2ndgreenboy', 'left3boycoming', 'left3rdgirlbesideswhitepants', 'left3rdrunwaygirlbesideswhitepants', 'left4thgirlwithwhitepants', 'leftblkboy', 'leftbrowngirlfat', 'leftlightsweaterboy', 'leftorangeboy', 'leftrunningboy', 'leftshortgirl', 'lightredboy', 'manaftercars', 'manatwhiteright', 'manbesideslight', 'manfarbesidespool', 'manleftsmallwhitehouse', 'manrun', 'manupstairs', 'midblkboyplayingphone', 'midboyplayingphone', 'midwhitegirl', 'moto2', 'nightboy', 'nightrightboy1', 'orangegirl', 'pinkgirl', 'redgirl', 'redgirlafterwhitecar', 'redgirlsits', 'redlittleboy', 'right2ndblkboy', 'right4thboy', 'rightbiggreenboy', 'rightblkboy', 'rightblkboy2', 'rightblkboybesidesred', 'rightblkgirl', 'rightboy479', 'rightboyatwindow', 'rightboybesidesredcar', 'rightboystand', 'rightboywithbluebackpack', 'rightboywithwhite', 'rightestblkboy', 'rightestblkboy2', 'rightgreenboy', 'rightofth3boys', 'rightredboy', 'rightredboy1227', 'rightredboy954', 'rightwhitegirl', 'rightwhitegirlleftpink', 'runninggreenboyafterwhite', 'singleboywalking', 'sitleftboy', 'the2ndboyunderbasket', 'the4thboy', 'the4thboystandby', 'theleftboytakingball', 'theleftestrunningboy', 'trolleywith2boxes1_quezhen', 'waiterontheothersideofwindow', 'whiteboy1_quezhen', 'whitegirl', 'whitegirl1227', 'whitegirl2', 'whitegirl2right', 'whitegirlcoming', 'yellowgirlwithbowl', '2boys', '2boysup', '2girlgoleft', '2rdcarcome', '2rdtribike', '2up', '3bike2', '3blackboys', '3girl1', '3whitemen', '4boys2left', '4sisters', 'agirl', 'basketman', 'bigbus', 'bike150', 'bikeboygo', 'bikeboyleft', 'bikeinhand', 'blackbag', 'blackbaggirl', 'blackboy256', 'blackcar', 'blackinpeople', 'blackman2', 'blackpantsman', 'blackphoneboy', 'blackturnr', 'bluecar', 'boybehindbus', 'boyleft161', 'boyright', 'boysback', 'boyscome', 'boyturn', 'browncarturn', 'carclosedoor', 'carfarstart', 'cargirl2', 'carleaveturnleft', 'carstart', 'carstop', 'carturn', 'carturnleft', 'checkedshirt', 'comecar', 'cycleman', 'downmoto', 'easy_4women', 'easy_blackboy', 'etrike', 'girl2trees', 'girlbike', 'girlcoat', 'girlleaveboys', 'girlleft2right1', 'girlleft2right2', 'girlpickboy', 'girlumbrella', 'leftblackboy', 'man', 'manaftercar', 'mancrossroad', 'manglass1', 'manglass2', 'manphone', 'manwait1', 'man_head', 'man_with_black_clothes2', 'man_with_black_clothes3', 'minibus125', 'motobike', 'motocross', 'motoman', 'occludedmoto', 'rightbluegirl', 'threepeople', 'treeboy', 'twopeople', 'umbregirl', 'unbrellainbike', 'whiteblcakwoman', 'whiteboycome', 'whitecar', 'whiteman', 'whiteshirt', 'whitewoman', 'woman', 'womanopendoor', '2boysatblkcarend', '2boysbesidesblkcar', '2boyscome', '2boyscome245', '2girlinrain', '2girlsridebikes', '2gointrees', '2ndbus', '2ndcarcome', '2ndgirlmove', '2outdark', '2sisiters', '3rdboy', '4boysbesidesblkcar', 'ab_boyfromtrees', 'ab_girlrideintrees', 'ab_moto2north0', 'ab_motocometurn', 'basketball', 'basketboyblack', 'basketboywhite', 'bike2north', 'bikeblkbag', 'bikeblkturn', 'bikeboy173', 'bikeboycome', 'bikeboystrong', 'bikecoming', 'biked', 'bikefromnorth', 'bikefromnorth2', 'bikefromnorth257', 'bikeorange', 'biketonorth', 'biketurn', 'bikeumbrellacome', 'bikewithbag', 'blackaftertrees', 'blackbagbike', 'blackboypushbike', 'blackcar126', 'blackcar131', 'blackcarcome', 'blackcargo', 'blackcarturn175', 'blackcarturn183', 'blackmanleft', 'blackof4bikes', 'blackridebike', 'blackridebike2', 'blacktallman', 'blkbikefromnorth', 'blkboyatbike', 'blkboyback636', 'blkboyonleft', 'blkboywithblkbag', 'blkboywithumbrella', 'blkcar2north', 'blkcarcome', 'blkcarcome155', 'blkcarcomeinrain', 'blkcarfollowingwhite', 'blkcargo', 'blkcarinrain', 'blkgirlbike', 'blkmaninrain', 'blkmoto', 'blkmotocome', 'blueboy85', 'blueboybike', 'blueboycome', 'blueboywalking', 'bluegirl', 'bluelittletruck', 'bluemanatbike', 'blueumbrellagirl', 'boyalone', 'boybesidesblkcarrunning', 'boybesidescarwithouthat', 'boybetween2blkcar', 'boybikeblueumbrella', 'boybikewithbag', 'boyblackback', 'boycome', 'boydown', 'boyfromdark2', 'boygointrees', 'boyleave', 'boyouttrees', 'boyride2trees', 'boyrideoutandin', 'boyridesbike', 'boyrun', 'boyshead', 'boyshead2', 'boyshorts', 'boysitbike', 'boysumbrella', 'boysumbrella2', 'boysumbrella3', 'boytakebox', 'boytakepath', 'boytakesuicase', 'boyumbrella4', 'boywithshorts', 'boywithshorts2', 'bus', 'bus2', 'bus2north', 'car2north', 'car2north2', 'car2north3', 'carbesidesmoto', 'carcomeonlight2', 'carfromnorth', 'carfromnorth2', 'carlight', 'carout', 'carstart2east', 'darkgiratbike', 'dogfollowinggirl', 'doginrain', 'dogouttrees', 'drillmaster', 'e-tribike', 'e-tricycle', 'farredcar', 'farwhiteboy', 'fatmancome', 'folddenumbrellainhand', 'girlaftertree', 'girlalone', 'girlbesidesboy', 'girlbike156', 'girlbikeinlight', 'girlblack', 'girlfoldumbrella', 'girlgoleft', 'girlridesbike', 'girltakebag', 'girltakemoto', 'goaftrtrees', 'gonemoto_ab', 'greenboywithgirl', 'greengirls', 'guardatbike_ab', 'huggirl', 'jeepblack', 'jeepleave', 'leftblkboy648', 'leftboy', 'leftgirl', 'leftof2girls', 'lightcarcome', 'lightcarfromnorth', 'lightcarstart', 'lightcarstop', 'lonelyman', 'man2startmoto', 'manaftercar114', 'manaftertrees', 'manatmoto', 'manbikecoming', 'mancarstart', 'manfromcar', 'manfromcar302', 'maninfrontofbus', 'manopendoor', 'manrun250', 'manstarttreecar', 'mengointrees', 'midboy', 'midgirl', 'minibus152', 'minibuscome', 'moto2north', 'moto2north1', 'moto2north2', 'moto2trees', 'moto2trees2', 'moto78', 'motobesidescar', 'motocome', 'motocome122', 'motocomenight', 'motofromdark', 'motoinrain', 'motolight', 'motoprecede', 'motoslow', 'motosmall', 'mototake2boys', 'mototurn', 'mototurn102', 'mototurn134', 'mototurnleft', 'mototurnright', 'motowithblack', 'motowithgood', 'motowithtopcoming', 'nikeatbike', 'oldwoman', 'pinkbikeboy', 'raincarstop', 'rainyboyaftertrees', 'rainysuitcase', 'rainywhitecar', 'redboywithblkumbrella', 'redcar', 'redcarturn', 'redmotocome', 'redshirtman', 'redup', 'redwhitegirl', 'rightblkboy188', 'rightgirl', 'rightgirlbike', 'schoolbus', 'silvercarcome', 'sisterswithbags', 'stripeman', 'stronggirl', 'stubesideswhitecar', 'suitcase', 'take-out-motocoming', 'takeoutman', 'takeoutman953', 'takeoutmanleave', 'takeoutmoto', 'tallboyblack', 'tallwhiteboy', 'the4thwhiteboy', 'trashtruck', 'truck', 'truckcoming', 'truckk', 'truckwhite', 'umbellaatnight', 'umbrellabikegirl', 'umbrellainblack', 'umbrellainred', 'umbrellainyellowhand', 'whiteaftertree', 'whiteaftertrees', 'whiteatright', 'whiteboy', 'whiteboy395', 'whiteboyatbike', 'whiteboybike', 'whiteboycome598', 'whiteboyphone', 'whiteboyright', 'whiteboywait', 'whiteboywithbag', 'whitecar70', 'whitecarafterbike', 'whitecarcome', 'whitecarcome192', 'whitecarcomes', 'whitecarcoming', 'whitecarfromnorth', 'whitecargo', 'whitecarinrain', 'whitecarleave', 'whitecarleave198', 'whitecarstart', 'whitecarstart126', 'whitecarturn', 'whitecarturn2', 'whitecarturn85', 'whitecarturn137', 'whitecarturn178', 'whitecarturnl', 'whitecarturnl2', 'whitedown', 'whitegirl209', 'whiteskirtgirl', 'whitesuvcome', 'whiteTboy', 'womanaroundcar', 'womanongrass', 'yellowgirl', 'yellowumbrellagirl', '2ndbikecoming', 'ab_bikeboycoming', 'ab_minibusstops', 'ab_motoinrain', 'ab_mototurn', 'ab_redboyatbike', 'ab_shorthairgirlbike', 'bike2trees86', 'bikecome', 'bikecoming176', 'bikefromwest', 'bikeout', 'biketurndark', 'biketurnleft', 'biketurnleft2', 'blackboy186', 'blackcarback', 'blackcarcoming', 'blkbikecomes', 'blkboy198', 'blkboybike', 'blkcarcome115', 'blkcarinrain107', 'blkcarstart', 'blkman2trees', 'blkmototurn', 'blkridesbike', 'blkskirtwoman', 'blktakeoutmoto', 'bluebike', 'blueboyopenbike', 'bluemanof3', 'bluemoto', 'bluetruck', 'boycomingwithumbrella', 'boymototakesgirl', 'boyridesbesidesgirl', 'boywithblkbackpack', 'boywithumbrella', 'browncar2east', 'browncar2north', 'bus2north111', 'camonflageatbike', 'carstart189', 'carstarts', 'carturncome', 'carturnleft109', 'comebike', 'darkredcarturn', 'dogunderthelamp', 'farwhitecarturn', 'girlinfrontofcars', 'girlintrees', 'girlplayingphone', 'girlshakeinrain', 'girltakingmoto', 'girlthroughtrees', 'girlturnbike', 'girlwithblkbag', 'girlwithumbrella', 'greenboy438', 'guardman', 'leftgirlafterlamppost', 'leftgirlunderthelamp', 'leftwhitebike', 'lightmotocoming', 'manafetrtrees', 'mantoground', 'manwalkincars', 'manwithyellowumbrella', 'meituanbike', 'meituanbike2', 'midblkbike', 'minibusback', 'moto2ground', 'moto2north101', 'moto2west', 'motocome2left', 'motocomeinrain', 'motocometurn', 'motocoming', 'motocominginlight', 'motoinrain56', 'motolightturnright', 'motostraught2east', 'mototake2boys123', 'mototaking2boys', 'mototakinggirl', 'mototurntous', 'motowithtop', 'nearmangotoD', 'nightmototurn', 'openningumbrella', 'orangegirlwithumbrella', 'pinkgirl285', 'rainblackcarcome', 'raincarstop2', 'redbaginbike', 'redgirl2trees', 'redminirtruck', 'redumbrellagirlcome', 'rightboywitjbag', 'rightgirlatbike', 'rightgirlbikecome', 'rightgirlwithumbrella', 'rightof2boys', 'runningwhiteboy', 'shunfengtribike', 'skirtwoman', 'smallmoto', 'takeoutmoto521', 'takeoutmototurn', 'trimototurn', 'turnblkbike', 'whitebikebehind', 'whitebikebehind172', 'whitebikebehind2', 'whiteboyback', 'whitecar2west', 'whitecarback', 'whitecarstart183', 'whitecarturnright248', 'whitegirlatbike', 'whitegirlcrossingroad', 'whitegirlundertheumbrella', 'whitegirlwithumbrella', 'whitemancome', 'whiteminibus197', 'whitemoto', 'whitemotoout', 'whiteof2boys', 'whitesuvstop', 'womanstartbike', 'yellowatright', 'yellowcar', 'yellowtruck', '10crosswhite', '10phone_boy', '10rightblackboy', '10rightboy', '11righttwoboy', '11runone', '11runthree', '1boygo', '1handsth', '1phoneblue', '1rightboy', '1righttwogreen', '1whiteteacher', '2runfive', '2runfour', '2runone', '2runsix', '2runtwo', '2whitegirl', '4four', '4one', '4runeight', '4runone', '4thgirl', '4three', '4two', '5manwakeright', '5numone', '5one', '5runfour', '5runone', '5runthree', '5runtwo', '5two', '6walkgirl', '7one', '7rightblueboy', '7rightredboy', '7rightwhitegirl', '7runone', '7runthree', '7runtwo', '7two', '8lastone', '9handlowboy', '9hatboy', '9whitegirl', 'abeauty_1202', 'aboyleft_1202', 'aboy_1202', 'ab_bolster', 'ab_catescapes', 'ab_hyalinepaperatground', 'ab_leftfoam', 'ab_leftmirrordancing', 'ab_pingpongball', 'ab_pingpongball3', 'ab_rightcupcoming_infwhite_quezhen', 'ab_righthandfoamboard', 'ab_rightmirror', 'actor_1202', 'agirl1_1202', 'agirl_1202', 'battlerightblack', 'blackbetweengreenandorange', 'blackdownball', 'blackdresswithwhitefar', 'blackman_0115', 'blackthree_1227', 'blklittlebag', 'blkumbrella', 'Blue_in_line_1202', 'bolster', 'bolster_infwhite', 'bookatfloor', 'boy2_0115', 'boy2_1227', 'boy_0109', 'boy_0115', 'boy_1227', 'cameraman_1202', 'camera_1202', 'catbrown', 'dogforward', 'dotat43', 'downwhite_1227', 'elector_0115', 'elector_1227', 'exercisebook', 'fallenbikeitself', 'foamboardatlefthand', 'folderatlefthand', 'folderinrighthand', 'foundsecondpeople_0109', 'frontmirror', 'girlafterglassdoor2', 'girlatwindow', 'girlback', 'girloutreading', 'girlrightcomein', 'greenfaceback', 'greenleftbackblack', 'greenleftthewhite', 'greenrightblack', 'higherthwartbottle_quezhen', 'hyalinepaperfrontclothes', 'left4thblkboyback', 'leftbottle', 'leftclosedexersicebook', 'leftcup', 'leftfallenchair_inf_white', 'lefthandfoamboard', 'lefthyalinepaper', 'leftmirror2', 'leftmirrorshining', 'leftredcup', 'leftthrowfoam', 'left_first_0109', 'left_two_0109', 'lovers_1227', 'lover_1202', 'lowerfoam2throw', 'man_0109', 'midpinkblkglasscup', 'mirroratleft', 'mirrorfront', 'nearestleftblack', 'openthisexersicebook', 'orange', 'othersideoftheriver_1227', 'outer2leftmirrorback', 'outerfoam', 'peoplefromright_0109', 'pickuptheyellowbook', 'pingpongball', 'pingpongball2', 'pingpongpad', 'pingpongpad2', 'redcupatleft', 'right2ndblkpantsboy', 'rightbackcup', 'rightbattle', 'rightbottle', 'rightboy_1227', 'rightexercisebookwillfly', 'rightgreen', 'righthand`sfoam', 'righthunchblack', 'rightmirrorbackwards', 'rightmirrorlikesky', 'rightmirrornotshining', 'rightof2cupsattached', 'rightredcup_quezhen', 'rightshiningmirror', 'rightstripeblack', 'rightumbrella_quezhen', 'rightwhite_1227', 'shotmaker', 'shotmaker2', 'swan2_0109', 'Take_an_umbrella_1202', 'thefirstexcersicebook', 'The_girl_back_at_the_lab_1202', 'The_girl_with_the_cup_1202', 'The_one_on_the_left_in_black_1202', 'twopeople_0109', 'twoperpson_1202', 'twoperson_1202', 'two_1227', 'wanderingly_1202', 'whitacatfrombush', 'whitebetweenblackandblue', 'whiteboy242', 'whitecat', 'whitecatjump', 'whitegirl2_0115', 'whitegirl_0115', 'whitewoman_1202', 'yellowexcesicebook', 'Aab_whitecarturn', 'Ablkboybike77', 'Abluemotoinrain', 'Aboydownbike', 'Acarlightcome', 'Agirlrideback', 'Ahercarstart', 'Amidredgirl', 'Amotoinrain150', 'Amotowithbluetop', 'AQbikeback', 'AQblkgirlbike', 'AQboywithumbrella415', 'AQgirlbiketurns', 'AQmanfromdarktrees', 'AQmidof3boys', 'AQmotomove', 'AQraincarturn2', 'AQrightofcomingmotos', 'AQtaxi', 'AQwhiteminibus', 'AQwhitetruck', 'Aredtopmoto', 'Athe3rdboybesidescar', 'Awhitecargo', 'Awhiteleftbike', 'Awoman2openthecardoor', '1rowleft2ndgirl', '1strow3rdboymid', '1strowleft3rdgirl', '1strowright1stboy', '1strowright2ndgirl', '1strowrightgirl', '2ndboyfarintheforest2right', 'backpackboyhead', 'basketballatboysrighthand', 'basketballbyNo_9boyplaying', 'basketballshooting', 'basketballshooting2', 'belowrightwhiteboy', 'belowyellow-gai', 'besom-ymm', 'besom2-gai', 'besom4', 'besom5-sq', 'besom6', 'blackruning', 'bord', 'bouheadupstream', 'boyalonecoming', 'boybesidesbar2putcup', 'boyfollowing', 'boyof2leaders', 'boyplayingphone366', 'boyshead509', 'boystandinglefttree', 'boyunderleftbar', 'boywithwhitebackpack', 'collegeofmaterial-gai', 'darkleftboy2left', 'elegirl', 'fardarkboyleftthe1stgirl', 'firstboythroughtrees', 'firstrightflagcoming', 'girl', 'girl2-gai', 'girloutqueuewithbackpack', 'girlshead', 'girlsheadwithhat', 'girlsquattingbesidesleftbar', 'girltakingblkumbrella', 'girl`sheadoncall', 'glassesboyhead', 'highright2ndboy', 'large2-gai', 'large3-gai', 'lastgirl-qzc', 'lastof4boys', 'lastrowrightboy', 'left11', 'left2flagfornews', 'left2ndgirl', 'left4throwboy', 'leftaloneboy-gai', 'leftbasketball', 'leftblkboyunderbasketballhoop', 'leftboy-gai', 'leftboyleftblkbackpack', 'leftbroom', 'leftconerbattle-gai', 'leftconergirl', 'leftdress-gai', 'leftdrillmasterstanding', 'leftdrillmasterstandsundertree', 'leftgirl1299', 'leftgirlat1row', 'leftgirlchecking', 'leftgirlchecking2', 'leftlastboy-sq', 'leftlastgirl-yxb', 'leftlastgirl2', 'leftmen-chong1', 'leftredflag-lsz', 'leftwhiteblack', 'left_leader', 'midboyblue', 'midflag-qzc', 'midtallboycoming', 'nearstrongboy', 'ninboy-gai', 'notebook-gai', 'redbackpackgirl', 'redbaggirlleftgreenbar', 'redgirl1497', 'right1stgirlin2ndqueue', 'right2nddrillmaster', 'right2ndfarboytakinglight2left', 'right2ndgirl', 'rightbhindbike-gai', 'rightblkgirlNo_11', 'rightblkgirlrunning', 'rightbottle-gai', 'rightbottle2-gai', 'rightboyleader', 'rightboywithbackpackandumbrella', 'rightboy`shead', 'rightdrillmasterunderthebar', 'rightfirstboy-ly', 'rightfirstgirl-gai', 'rightgirlplayingphone', 'rightholdball', 'rightholdball1096', 'rightof2boys953', 'rightof2cominggirls', 'rightofthe4girls', 'rightrunninglatterone', 'righttallholdball', 'righttallnine-gai', 'rightwhiteboy', 'runningwhiteboy249', 'schoolofeconomics-yxb', 'small2-gai', 'strongboy`head', 'tallboyNumber_9', 'toulan-ly', 'twoleft', 'twolinefirstone-gai', 'twopeopleelec-gai', 'tworightbehindboy-gai', 'whiteboy-gai', 'whiteboyup', 'whiteboy`head', 'whitehatgirl`sheadleftwhiteumbrella', 'whiteshoesleftbottle-gai']

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))


        self.attr=attr
        with open('/data1/Code/zhaziyi/OSTrack_modmiss_git/IPL/dataset_attr/LasHeR_Attributes/Attributes_order.txt') as f:
            attr_list = f.read().replace(' ','').split(',')
        if attr is not None and isinstance(attr, str):
            assert attr in attr_list
            attr_idx = attr_list.index(attr)

            for i in range(len(self.sequence_list)-1, -1, -1):
                seqname = self.sequence_list[i]
                fn = 'dataset_attr/LasHeR_Attributes/AttriSeqsTxt/' + seqname + '.txt'
                with open(fn) as f:
                    seq_attr = f.read().replace(' ','').split(',')
                if seq_attr[attr_idx]=='0':
                    del self.sequence_list[i]

        elif isinstance(attr, list) and len(attr)>0:
            attr_idxs = []
            for item in attr:
                assert item in attr_list
                attr_idxs.append(attr_list.index(item))
            for i in range(len(self.sequence_list)-1, -1, -1):
                seqname = self.sequence_list[i]
                fn = 'dataset_attr/LasHeR_Attributes/AttriSeqsTxt/' + seqname + '.txt'
                with open(fn) as f:
                    seq_attr = f.read().replace(' ','').split(',')
                if sum([int(seq_attr[idx]) for idx in attr_idxs])==0:
                    del self.sequence_list[i]

        print(f'Dataset: LasHeR_train, Attribute:{attr}, seq length: {len(self.sequence_list)}')

        
    def get_name(self):
        return 'LasHeR_trainingSet'

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, 'init.txt')
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    # def _get_sequence_path(self, seq_id):
    #     return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        #print('seq_id', seq_id)
        seq_name = self.sequence_list[seq_id]
        #print('seq_name', seq_name)
        seq_path = os.path.join(self.root, seq_name)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_v(self, seq_path, frame_id):
        frame_path_v = os.path.join(seq_path, 'visible', sorted([p for p in os.listdir(os.path.join(seq_path, 'visible')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        return self.image_loader(frame_path_v)
        
    def _get_frame_i(self, seq_path, frame_id):
        frame_path_i = os.path.join(seq_path, 'infrared', sorted([p for p in os.listdir(os.path.join(seq_path, 'infrared')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        return self.image_loader(frame_path_i)

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_name = self.sequence_list[seq_id]
        seq_path = os.path.join(self.root, seq_name)
        frame_list_v = [self._get_frame_v(seq_path, f) for f in frame_ids]
        frame_list_i = [self._get_frame_i(seq_path, f) for f in frame_ids]

        #print(frame_ids)
        #print(len(frame_list_v),len(frame_list_i),len(frame_list))
        #@print(len(frame_list_i))
        frame_list  = frame_list_v + frame_list_i # 6
        #print(len(frame_list))
        #exit()
        if seq_name not in self.sequence_list:
            print('warning!!!'*100)
        if anno is None:
            anno = self.get_sequence_info(seq_path)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        #return frame_list_v, frame_list_i, anno_frames, object_meta
        return frame_list, anno_frames, object_meta
