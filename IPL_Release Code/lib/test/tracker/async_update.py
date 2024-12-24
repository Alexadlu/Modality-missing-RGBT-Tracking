


class AsyncUpdate:
    def __init__(self, mode="once", thr = [1., 0., 0.], timing=None) -> None:
        """
        mode = once, always, timing
        """
        self.mode = mode
        self.update_log = []        # 1 is rgb, 2 is tir, 0 is no update
        self.score_thr = thr[0]
        self.rgb_thr = thr[1]
        self.tir_thr = thr[2]
        self.timing = timing

        self.score_log = []
        self.contr_rgb_log = []
        self.contr_tir_log = []

        self.lock_rgb = False
        self.lock_tir = False


    def __call__(self, score, contr_rgb, contr_tir) -> bool:
        if self.mode=="once":
            return self.once_update(score, contr_rgb, contr_tir)
        elif self.mode=="always":
            return self.always_update(score, contr_rgb, contr_tir)
        elif self.mode=="timing":
            return self.timing_update(score, contr_rgb, contr_tir)
        else:
            raise f"{self.mode} not exist."


    def _log(self, i):
        self.update_log.append(i)
        while len(self.update_log)>50:
            del self.update_log[0]
        return i


    def once_update(self, score, contr_rgb, contr_tir):
        """
        该模式下，每个模态只更新"一"次
        在always情况下连续5帧满足score条件但不更新，往后就都不更新
        """
        self.score_log.append(score)
        while len(self.score_log)>50:
            del self.score_log[0]
        if self.lock_rgb and self.lock_tir:
            return self._log(0)
        
        if score>=self.score_thr:
            gap=5
            if len(self.update_log)>=gap and min(self.score_log[-gap-1:-1])>self.score_thr:
                if 1 not in self.update_log[-gap:]:
                    self.lock_rgb=True
                if 2 not in self.update_log[-gap:]:
                    self.lock_tir=True
                    
            if (not self.lock_rgb) and contr_rgb<self.rgb_thr:
                return self._log(1)
            if (not self.lock_tir) and contr_tir<self.tir_thr:
                return self._log(2)
        return self._log(0)
        
        

    def always_update(self, score, contr_rgb, contr_tir):
        if score>self.score_thr:
            if contr_rgb<self.rgb_thr:
                return self._log(1)
            if contr_tir<self.tir_thr:
                return self._log(2)
            # if contr_rgb>=self.rgb_thr:
            #     return self._log(1)
            # if contr_tir>=self.tir_thr:
            #     return self._log(2)
            return self._log(0)


    def timing_update(self, score, contr_rgb, contr_tir) -> tuple:
        """
        定时更新，并且选择得分最高的，满足要求的更新
        """
        gap = self.timing
        self.score_log.append(score)
        while len(self.score_log)>50:
            del self.score_log[0]
        self.contr_rgb_log.append(contr_rgb)
        if len(self.contr_rgb_log)>20:
            del self.contr_rgb_log[0]
        self.contr_tir_log.append(contr_tir)
        if len(self.contr_tir_log)>20:
            del self.contr_tir_log[0]
        if (len(self.update_log)+1)%gap:
            return self._log(0), None
        
        max_score = max(self.score_log[-gap:])
        idx_reverse = self.score_log[-gap:].index(max_score) - gap
        contr_rgb = self.contr_rgb_log[idx_reverse]
        contr_tir = self.contr_tir_log[idx_reverse]

        if max_score>=self.score_thr:
            if contr_rgb<self.rgb_thr:
                return self._log(1), idx_reverse
            if contr_tir<self.tir_thr:
                return self._log(2), idx_reverse
            return self._log(0), None