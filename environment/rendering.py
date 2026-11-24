"""2D pygame visualisation of the KigaliCare environment (works headless via rgb_array)."""
from __future__ import annotations

import os
import numpy as np

W, H = 1100, 640
BG = (246, 244, 238)
INK = (35, 40, 48)
MUTED = (120, 126, 136)
GREEN, AMBER, RED, BLUE, GREY = (46, 160, 110), (232, 168, 40), (200, 60, 60), (52, 112, 190), (200, 200, 200)


class Renderer:
    def __init__(self, human: bool = True):
        if not human:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame
        self.pg = pygame
        pygame.init()
        pygame.font.init()
        self.human = human
        if human:
            self.screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("KigaliCare - care navigation environment")
        else:
            self.screen = pygame.Surface((W, H))
        self.f_big = pygame.font.SysFont("arial", 26, bold=True)
        self.f_med = pygame.font.SysFont("arial", 18, bold=True)
        self.f_sm = pygame.font.SysFont("arial", 15)
        self.clock = pygame.time.Clock()

    def _text(self, s, font, pos, color=INK):
        self.screen.blit(font.render(s, True, color), pos)

    def _bar(self, x, y, w, h, frac, color, label=None):
        pg = self.pg
        pg.draw.rect(self.screen, GREY, (x, y, w, h), border_radius=4)
        pg.draw.rect(self.screen, color, (x, y, int(w * float(np.clip(frac, 0, 1))), h), border_radius=4)
        if label:
            self._text(label, self.f_sm, (x, y - 18), MUTED)

    def draw(self, env):
        pg = self.pg
        from environment.custom_env import N_USERS, EPISODE_DAYS, SLOTS_PER_WEEK, REFERRALS_PER_MONTH
        s = self.screen
        s.fill(BG)
        self._text("KigaliCare  |  Kimironko pilot  |  Care-navigation coordinator", self.f_big, (24, 16))
        self._text(f"Day {env.day}/{EPISODE_DAYS}", self.f_med, (24, 56))
        self._bar(130, 60, 300, 14, env.day / EPISODE_DAYS, BLUE)
        self._text(f"Counsellor slots this week: {env.slots}/{SLOTS_PER_WEEK}", self.f_med, (470, 56))
        self._text(f"KMHRC referral places: {env.referrals}/{REFERRALS_PER_MONTH}", self.f_med, (800, 56))

        cw, gap, top = 200, 15, 110
        for i in range(N_USERS):
            x = 24 + i * (cw + gap)
            active = bool(env.active[i])
            pg.draw.rect(s, (255, 255, 255) if active else (228, 228, 228), (x, top, cw, 360), border_radius=12)
            pg.draw.rect(s, GREEN if active else MUTED, (x, top, cw, 360), 2, border_radius=12)
            self._text(f"User {i + 1}", self.f_med, (x + 12, top + 10))
            status = "ACTIVE" if active else "DROPPED"
            self._text(status, self.f_sm, (x + 100, top + 12), GREEN if active else RED)
            sev = float(env.sev[i])
            col = GREEN if sev < 0.4 else AMBER if sev < 0.7 else RED
            self._bar(x + 12, top + 60, cw - 24, 18, sev, col, "True severity (hidden)")
            ox = x + 12 + int((cw - 24) * float(env.obs_sev[i]))
            pg.draw.line(s, INK, (ox, top + 58), (ox, top + 82), 3)
            self._text("| = last assessed", self.f_sm, (x + 12, top + 86), MUTED)
            self._bar(x + 12, top + 140, cw - 24, 14, env.eng[i], BLUE, "Engagement")
            self._bar(x + 12, top + 190, cw - 24, 14, env.trust[i], (120, 90, 190), "Trust")
            self._text(f"Days since contact: {int(env.dsc[i])}", self.f_sm, (x + 12, top + 220))
            if env.appt[i]:
                self._text("Counsellor appt booked", self.f_sm, (x + 12, top + 250), GREEN)
            if getattr(env, "crisis_user", -1) == i and sev >= 0.95:
                self._text("CRISIS EVENT", self.f_med, (x + 12, top + 300), RED)
        self._text(f"Action: {env.last_action_text}", self.f_big, (24, 500))
        self._text(f"Step reward: {env.last_reward:+.2f}      Episode return: {env.total_reward:+.2f}", self.f_med, (24, 545))
        self._text("Goal: lower severity, keep users engaged, use scarce slots/referrals wisely, avoid crisis.", self.f_sm, (24, 585), MUTED)
        if self.human:
            pg.event.pump()
            pg.display.flip()
            self.clock.tick(env.metadata["render_fps"])
        return np.transpose(np.array(pg.surfarray.pixels3d(s)), (1, 0, 2)).copy()

    def close(self):
        if self.human:
            self.pg.display.quit()
        self.pg.quit()
