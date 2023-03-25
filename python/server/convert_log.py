"""
Todo:
  - 包（パオ）、などの特殊状況の処理
  - （暗槓時のaの位置と赤の処理について調べる）
"""
import json

BASE_URL ="https://tenhou.net/5/#json="
convert_id = lambda id: 50+id//36+1 if id in [16, 52, 88] else (id//36+1)*10+(id%36)//4+1
ten_name = {8000:"満貫",12000:"跳満",16000:"倍満",24000:"三倍満"}
yaku_dict = {0:"門前清自摸和",1:"立直",2:"一発",3:"槍槓",4:"嶺上開花",5:"海底摸月",6:"河底撈魚",7:"平和",8:"断幺九",9:"一盃口",10:"自風 東",11:"自風 南",12:"自風 西",13:"自風 北",14:"場風 東",15:"場風 南",16:"場風 西",17:"場風 北",18:"役牌 白",19:"役牌 發",20:"役牌 中",21:"両立直",22:"七対子",23:"混全帯幺九",24:"一気通貫",25:"三色同順",26:"三色同刻",27:"三槓子",28:"対々和",29:"三暗刻",30:"小三元",31:"混老頭",32:"二盃口",33:"純全帯幺九",34:"混一色",35:"清一色",36:"人和",37:"天和",38:"地和",39:"大三元",40:"四暗刻",41:"四暗刻単騎",42:"字一色",43:"緑一色",44:"清老頭",45:"九蓮宝燈",46:"純正九蓮宝燈",47:"国士無双",48:"国士無双１３面",49:"大四喜",50:"小四喜",51:"四槓子",52:"ドラ",53:"裏ドラ",54:"赤ドラ"}

class ConvertLog:
	"""
	半荘分のログを天鳳の形式に変換し、URLを生成する。
	add_log(self,obs_dict)
		obs_dict:一局分のログのdict（全員分）
		ログの追加（一局づつ）
	get_url(self)
		天鳳の牌譜URLを返す(str型)
	"""
	def __init__(self):
		self.logs = { "title":["",""], "name":["","","",""], "rule":{"disp":"","aka":1}, "log":[]}

	def add_log(self,obs_dict):
		log = [[0,0,0],[0,0,0,0],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
		for i,obs in enumerate(obs_dict.values()):
			obs=json.loads(obs.to_json())
			if i == 0:
				self.logs["name"] = obs["publicObservation"]["playerIds"]
				
				round = obs["publicObservation"]["initScore"]["round"] if "round" in obs["publicObservation"]["initScore"].keys() else 0
				log[0][0] = round

				if "honba" in obs["publicObservation"]["initScore"].keys():
					log[0][1] = obs["publicObservation"]["initScore"]["honba"]

				if "riichi" in obs["publicObservation"]["initScore"].keys():
					log[0][2] = obs["publicObservation"]["initScore"]["riichi"]

				log[1] = obs["publicObservation"]["initScore"]["tens"]

				for dora in obs["publicObservation"]["doraIndicators"]:
					log[2].append(convert_id(dora))

				if "noWinner" in obs["roundTerminal"].keys():
					log.append(["流局",obs["roundTerminal"]["noWinner"]["tenChanges"]])
				else:
					for win in obs["roundTerminal"]["wins"]:
						if "uraDoraIndicators" in win.keys():
							for uradora in win["uraDoraIndicators"]:
								log[3].append(convert_id(uradora))
						who_win = win["who"] if "who" in win.keys() else 0
						from_who_win = win["fromWho"] if "fromWho" in win.keys() else 0
						agari_info=[who_win,from_who_win,who_win]
						ten = win["ten"]
						magnification = 1
						if round%4 == who_win:
							magnification = 1.5
						adjusted_ten = int(ten//magnification)
						if adjusted_ten < 8000:
							fan = sum(win["fans"])
							fu = win["fu"]
							agari_info.append(f"{fan}飜{fu}符{ten}点")
						elif adjusted_ten < 32000:
							agari_info.append(f"{ten_name[int(adjusted_ten)]}{ten}点")
						else:
							agari_info.append(f"役満{ten}点")
						if "yakus" in win.keys():
							for i,yaku in enumerate(win["yakus"]):
								fan = win["fans"][i]
								if fan>0:
									agari_info.append(f"{yaku_dict[yaku]}({fan}飜)")
						else:
							for yaku in win["yakumans"]:
								agari_info.append(f"{yaku_dict[yaku]}(役満)")
						log.append(["和了",win["tenChanges"],agari_info])
			who=obs["who"] if "who" in obs.keys() else 0
			log[4+who*3] = [convert_id(id) for id in obs["privateObservation"]["initHand"]["closedTiles"]]
			tumo_count=0
			event_count=0
			while event_count<len(obs["publicObservation"]["events"]):
				event = obs["publicObservation"]["events"][event_count]
				who_event = event["who"] if "who" in event.keys() else 0
				if who == who_event:
					if not "type" in event.keys():
						tile = event["tile"] if "tile" in event.keys() else 0
						log[4+who*3+2].append(convert_id(tile))

					elif event["type"] == "EVENT_TYPE_TSUMOGIRI":
						log[4+who*3+2].append(60)

					elif event["type"] == "EVENT_TYPE_DRAW":
						log[4+who*3+1].append(convert_id(obs["privateObservation"]["drawHistory"][tumo_count]))
						tumo_count += 1

					elif event["type"] == "EVENT_TYPE_RIICHI":
						event_count+=1
						next_event=obs["publicObservation"]["events"][event_count]
						tile = next_event["tile"] if "tile" in next_event.keys() else 0
						log[4+who*3+2].append('r'+str(convert_id(tile)))


					elif event["type"] == "EVENT_TYPE_CHI":
						open = event["open"]
						chi_offset = [((0b0000000000011000 & open)>>3)%4,
		       						  ((0b0000000001100000 & open)>>5)%4,
									  ((0b0000000110000000 & open)>>7)%4,
		       						 ]
						chi_base_and_stolen = (0b1111110000000000 & open)>>10
						stolen = chi_base_and_stolen%3
						chi_base = ((chi_base_and_stolen//3)%7)*4+((chi_base_and_stolen//3)//7)*36
						open_tile = [chi_base+stolen*4+chi_offset[stolen],
		   							 min(chi_base+((stolen+1)%3)*4+chi_offset[((stolen+1)%3)],chi_base+((stolen+2)%3)*4+chi_offset[((stolen+2)%3)]),
									 max(chi_base+((stolen+1)%3)*4+chi_offset[((stolen+1)%3)],chi_base+((stolen+2)%3)*4+chi_offset[((stolen+2)%3)])
									]
						log[4+who*3+1].append('c'+ str(convert_id(open_tile[0])) + str(convert_id(open_tile[1])) + str(convert_id(open_tile[2])))

					elif event["type"] == "EVENT_TYPE_PON":
						open = event["open"]
						mask_from = open%4
						pon_unused_offset = ((0b0000000001100000 & open)>>5)%4
						pon_base_and_stolen = (0b1111111000000000 & open)>>9
						pon_base = (pon_base_and_stolen//3)*4
						stolen = pon_base_and_stolen%3
						open_tile = list(range(pon_base,pon_base+4))
						open_tile.pop(pon_unused_offset)
						stolen_tile = open_tile.pop(stolen)
						open_tile = [str(convert_id(id)) for id in open_tile]
						open_tile.insert(3-mask_from,'p'+str(convert_id(stolen_tile)))
						log[4+who*3+1].append(''.join(open_tile))

					elif event["type"]=="EVENT_TYPE_CLOSED_KAN":
						open = event["open"] if "open" in event.keys() else 0
						kan_tile = open>>8
						open_tile=[str(convert_id((kan_tile//4)*4+1)),
		 						   str(convert_id((kan_tile//4)*4)),
								   str(convert_id((kan_tile//4)*4+2)),
								   'a',
								   str(convert_id((kan_tile//4)*4+3))
								  ]
						log[4+who*3+2].append(''.join(open_tile))
					
					elif event["type"]=="EVENT_TYPE_ADDED_KAN":
						open = event["open"]
						mask_from = open%4
						pon_unused_offset = ((0b0000000001100000 & open)>>5)%4
						pon_base_and_stolen = (0b1111111000000000 & open)>>9
						pon_base = (pon_base_and_stolen//3)*4
						stolen = pon_base_and_stolen%3
						open_tile = list(range(pon_base,pon_base+4))
						added_tile=open_tile.pop(pon_unused_offset)
						stolen_tile = open_tile.pop(stolen)
						open_tile.append(stolen_tile)
						open_tile = [str(convert_id(id)) for id in open_tile]
						open_tile.insert(3-mask_from,'k'+str(convert_id(added_tile)))
						log[4+who*3+2].append(''.join(open_tile))

					elif event["type"]=="EVENT_TYPE_OPEN_KAN":
						open = event["open"]
						mask_from = open%4
						kan_tile = open>>8
						open_tile = list(range((kan_tile//4)*4,(kan_tile//4)*4+4))
						stolen_tile = open_tile.pop(kan_tile%4)
						open_tile = [str(convert_id(id)) for id in open_tile]
						insert_pos = 3-mask_from if mask_from>=2 else 3
						open_tile.insert(insert_pos,'m'+str(convert_id(stolen_tile)))
						log[4+who*3+1].append(''.join(open_tile))
						log[4+who*3+2].append(0)
				
				event_count+=1						

		self.logs["log"].append(log)
	
	def get_url(self):
		return BASE_URL+json.dumps(self.logs, separators=(',', ':'),ensure_ascii=False)