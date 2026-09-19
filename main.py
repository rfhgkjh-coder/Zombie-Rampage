"""
2D Top-Down Shooter — FastAPI + WebSockets backend.

Run with:
    uvicorn main:app --reload

Serves static/index.html at "/" and drives an authoritative game loop
per room over a single WebSocket endpoint:
    /ws/{room_id}/{nickname}?client_id=...
followed immediately by a first client message {"type": "auth", "token":
..., "password": ...} — the token and password are secrets and are never
put in the URL (see the auth-message note in ws_endpoint for why).

client_id is a stable id the browser keeps in sessionStorage, used to let a
player reclaim their slot (and host rights) after a page reload or a brief
disconnect, instead of losing them to a freshly-picked host. client_id
alone is NOT trusted for this, though — it's visible to anyone on the same
LAN sniffing the (unencrypted, plain ws://) connection, or just guessable.
The server also hands out a random secret `token` the first time a player
joins; reclaiming a slot requires presenting the matching token, so knowing
someone else's client_id alone can't steal their slot, stats, or host
rights. (Running behind TLS/wss in production is still recommended so the
token itself isn't sniffable either.)

Every nickname is now a real password-protected account (see
claim_or_verify_account / accounts.json): the first successful login for a
nickname registers the password, every login after that must match it.
This also lets the server enforce ACTIVE_SESSIONS — one account can only
hold a single connected/reconnect-pending slot at a time anywhere on the
server, so the same account can't be played from two tabs or two devices
at once. A same-device page reload still reconnects into the same slot via
client_id+token without needing the password again.
"""

import asyncio
import hashlib
import json
import math
import os
import random
import secrets
import time
import uuid
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Top-Down Shooter")

# Lets a game page hosted on itch.io (served from *.itch.zone, a
# different origin than this server) talk to this backend. Harmless to
# leave wide open here — there's nothing behind these routes a browser
# could abuse cross-origin (no cookies/session auth, no destructive
# unauthenticated actions), and the WebSocket handshake itself isn't
# gated by CORS the way fetch()/XHR are, so this mainly matters if you
# ever add plain HTTP endpoints.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

MAP_W, MAP_H = 3000, 2000
TICK_RATE = 20
DT = 1.0 / TICK_RATE
PLAYER_SPEED = 260.0
PLAYER_RADIUS = 20
ZOMBIE_RADIUS = 20
BULLET_RADIUS = 5
RESPAWN_TIME = 3.0
MAX_LOOT = 8
LOOT_SPAWN_INTERVAL = 6.0
LOOT_LIFETIME = 45.0  # uncollected loot despawns so the MAX_LOOT cap doesn't
                       # permanently fill up with drops nobody walked past
WAVE_BREAK_TIME = 4.0
RECONNECT_GRACE = 20.0  # seconds a disconnected player's slot (and host rights) is held

# Match-end rules
PVP_TIME_LIMIT = 240.0   # PvP ends after this many seconds...
PVP_KILL_LIMIT = 15      # ...or once someone hits this many kills, whichever first
BOSS_WAVE_INTERVAL = 5   # a boss zombie joins every 5th wave

# PvE is an endless survival mode now — there is no "final" wave to clear
# and win. HARD_RAMP_WAVE is just the point where difficulty stops growing
# gently and starts climbing steeply, so a run that's gone on a while
# actually starts to hurt.
HARD_RAMP_WAVE = 10

LIVES_MAX = 3            # each death costs one life; 0 left = out for the rest of the run
TOXIC_ZONE_RADIUS = 85   # a toxic zombie's damage aura — standing inside it hurts even without it touching you
TOXIC_ZONE_DPS = 14      # damage per second while inside that aura
MAX_ZOMBIES_PER_WAVE = 90  # hard ceiling so an endless run's horde size can't grow forever

TROPHIES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trophies.json")

WEAPONS = {
    "pistol": {
        "name": "Pistol",
        "mag": 12,
        "reserve_max": 72,
        "dmg": 14,
        "fire_rate": 0.28,
        "speed": 1000,
        "spread": 0.03,
        "pellets": 1,
        "reload_time": 1.2,
        "infinite_reserve": True,  # sidearm never truly runs dry — otherwise
                                    # a long PvE run with no ammo pickups in
                                    # reach just goes silent
    },
    "rifle": {
        "name": "Rifle",
        "mag": 30,
        "reserve_max": 150,
        "dmg": 16,
        "fire_rate": 0.1,
        "speed": 1200,
        "spread": 0.06,
        "pellets": 1,
        "reload_time": 1.8,
    },
    "shotgun": {
        "name": "Shotgun",
        "mag": 6,
        "reserve_max": 36,
        "dmg": 9,
        "fire_rate": 0.75,
        "speed": 950,
        "spread": 0.32,
        "pellets": 7,
        "reload_time": 2.2,
    },
}

PLAYER_COLORS = [
    "#5EEAD4", "#FFB703", "#F97066", "#7C9CFF",
    "#C084FC", "#4ADE80", "#F472B6", "#38BDF8",
]

ZOMBIE_STATS = {
    "normal": {"hp": 60, "speed": 90, "dmg": 10, "radius": 20},
    "fast": {"hp": 32, "speed": 165, "dmg": 6, "radius": 16},
    "tank": {"hp": 220, "speed": 52, "dmg": 18, "radius": 27},
    "toxic": {"hp": 85, "speed": 82, "dmg": 15, "radius": 22},
    "hunter": {"hp": 48, "speed": 205, "dmg": 12, "radius": 16},
    "boss": {"hp": 700, "speed": 70, "dmg": 22, "radius": 42},
}


def new_id() -> str:
    return uuid.uuid4().hex[:10]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def dist_point_to_segment(px, py, ax, ay, bx, by):
    """Distance from (px,py) to the closest point on segment (ax,ay)-(bx,by).
    Used so a fast bullet's hit test covers the whole path it swept through
    this tick, not just where it happened to land — a bullet doing 1200px/s
    covers 60px per 50ms tick, which is more than wide enough to skip clean
    over a 20-40px-radius target if we only checked its endpoint."""
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return dist(px, py, ax, ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return dist(px, py, ax + t * dx, ay + t * dy)


def difficulty_scale(wave: int) -> float:
    """1.0 through HARD_RAMP_WAVE, then climbs steeply — this is what makes
    the run actually start to hurt once it's gone on a while, instead of
    the gentle +2-zombies-per-wave growth staying survivable forever."""
    if wave <= HARD_RAMP_WAVE:
        return 1.0
    extra = wave - HARD_RAMP_WAVE
    return 1.0 + extra * 0.22


ZOMBIE_GRID_CELL = 160.0  # bigger than a bullet's max per-tick travel (~60px)
                          # plus the largest zombie radius, so a 3x3
                          # neighborhood around a bullet's cell always
                          # covers everything its swept segment could touch


def _grid_cell(x, y):
    return int(x // ZOMBIE_GRID_CELL), int(y // ZOMBIE_GRID_CELL)


def build_zombie_grid(zombies):
    """Bucket zombie indices by grid cell once per tick, so bullet
    collision only has to test the handful of zombies near it instead of
    every zombie in the room — this is what keeps O(bullets x zombies)
    from becoming the bottleneck once a late wave has 60-90 zombies alive
    and several players shooting shotguns. Rebuilt fresh every tick rather
    than kept incrementally in sync, since zombies only move a few px/tick
    anyway and this is O(zombies) either way."""
    grid: Dict[tuple, list] = {}
    for idx, z in enumerate(zombies):
        if z["hp"] <= 0:
            continue
        grid.setdefault(_grid_cell(z["x"], z["y"]), []).append(idx)
    return grid


def nearby_zombie_indices(grid, x, y):
    cx, cy = _grid_cell(x, y)
    out = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            bucket = grid.get((cx + dx, cy + dy))
            if bucket:
                out.extend(bucket)
    return out


def random_spawn():
    return random.uniform(200, MAP_W - 200), random.uniform(200, MAP_H - 200)


def random_edge_point():
    side = random.randint(0, 3)
    if side == 0:
        return random.uniform(0, MAP_W), 20
    if side == 1:
        return random.uniform(0, MAP_W), MAP_H - 20
    if side == 2:
        return 20, random.uniform(0, MAP_H)
    return MAP_W - 20, random.uniform(0, MAP_H)


# --------------------------------------------------------------------------
# Trophies — a simple persistent "currency" (like cups in a Brawl-Stars-style
# game) keyed by nickname and stored in a flat JSON file.
# --------------------------------------------------------------------------

def _load_trophies() -> dict:
    try:
        with open(TROPHIES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


_TROPHIES: dict = _load_trophies()


def _save_trophies():
    try:
        # Snapshot with dict(...) before handing off to the worker thread:
        # _TROPHIES can still be mutated by another room's end_match() on
        # the event-loop thread while this thread is mid-json.dump, and
        # dumping a dict that's being resized out from under you raises
        # RuntimeError("dictionary changed size during iteration").
        with open(TROPHIES_FILE, "w", encoding="utf-8") as f:
            json.dump(dict(_TROPHIES), f)
    except Exception:
        pass


def get_trophies(nickname: str) -> int:
    return _TROPHIES.get(nickname.strip().lower(), 0)


def add_trophies(nickname: str, delta: int) -> int:
    """Update the in-memory total only — does NOT touch disk. Callers that
    change several players' trophies (i.e. end_match) should call
    save_trophies_async() once after the whole batch, not per player,
    otherwise a match with N players blocks the event loop N times."""
    key = nickname.strip().lower()
    total = max(0, _TROPHIES.get(key, 0) + delta)
    _TROPHIES[key] = total
    return total


async def save_trophies_async():
    # json.dump() to disk is blocking I/O; running it in a worker thread
    # keeps it from stalling every other room's game loop on this event
    # loop while the write happens.
    await asyncio.to_thread(_save_trophies)


# --------------------------------------------------------------------------
# Accounts — every nickname must be registered with a password, so the
# trophy total (and, since ACTIVE_SESSIONS below, the single play-slot)
# attached to it can never be used by whoever else types the same name.
# This is NOT a real auth system (no sessions, no email recovery, salted
# SHA-256 rather than a proper KDF) — it's just enough to stop nickname
# squatting/collisions in a game played with friends.
# --------------------------------------------------------------------------

ACCOUNTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accounts.json")
MIN_PASSWORD_LEN = 4


def _load_accounts() -> dict:
    try:
        with open(ACCOUNTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


_ACCOUNTS: dict = _load_accounts()


def _save_accounts():
    try:
        with open(ACCOUNTS_FILE, "w", encoding="utf-8") as f:
            json.dump(dict(_ACCOUNTS), f)
    except Exception:
        pass


async def save_accounts_async():
    await asyncio.to_thread(_save_accounts)


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


async def claim_or_verify_account(nickname: str, password: str):
    """Called only when minting a brand-new identity (never on a
    token-verified reconnect — that already proves who you are).

    Returns (ok, error_message). A password is now required for every
    nickname: the first successful login for a given nickname registers
    that password on the spot (claims it for next time); every login after
    that must supply the matching password. There is no more "open"
    nickname — this is what lets ACTIVE_SESSIONS below treat a nickname as
    a real account and enforce a single simultaneous session for it."""
    key = nickname.strip().lower()
    password = (password or "").strip()

    if not password:
        return False, "Enter a password — it is now required for every nickname."
    if len(password) < MIN_PASSWORD_LEN:
        return False, f"Password must be at least {MIN_PASSWORD_LEN} characters."

    acct = _ACCOUNTS.get(key)

    if acct is None:
        salt = secrets.token_hex(8)
        _ACCOUNTS[key] = {"salt": salt, "hash": _hash_password(password, salt)}
        await save_accounts_async()
        return True, None

    if _hash_password(password, acct["salt"]) == acct["hash"]:
        return True, None

    return False, "Incorrect password for this nickname."


# --------------------------------------------------------------------------
# Active sessions — since every nickname is now a real password-protected
# account, one account should never be playing from two places (two tabs,
# two devices) at once. This maps account key -> (room_id, player_id) of
# the single slot currently allowed to hold it. A slot keeps the claim for
# as long as it exists in its room — connected and mid-match, or merely
# disconnected-and-in-grace, both count as "still holding it" — and only
# gives it up when the slot itself is removed (explicit leave, or the
# reconnect grace period expiring). A same-device reload reconnects into
# that same slot via client_id+token and never touches this map.
# --------------------------------------------------------------------------

ACTIVE_SESSIONS: Dict[str, tuple] = {}


def session_conflict(key: str) -> bool:
    """True if some OTHER still-existing slot already holds this account."""
    owner = ACTIVE_SESSIONS.get(key)
    if owner is None:
        return False
    owner_room_id, owner_pid = owner
    owner_room = ROOMS.get(owner_room_id)
    if owner_room is None or owner_pid not in owner_room.players:
        # Stale entry — the slot that made this claim is long gone (e.g.
        # the process behind a room crashed before it could release it).
        # Self-heal instead of permanently locking the account out.
        ACTIVE_SESSIONS.pop(key, None)
        return False
    return True


def claim_session(key: str, room_id: str, pid: str):
    ACTIVE_SESSIONS[key] = (room_id, pid)


def release_session(key: str, room_id: str, pid: str):
    # Only clear it if it still points at THIS slot — guards against a
    # late release (e.g. a delayed grace-period cleanup) wiping out a
    # legitimate newer claim the account has since made elsewhere.
    if ACTIVE_SESSIONS.get(key) == (room_id, pid):
        ACTIVE_SESSIONS.pop(key, None)



# --------------------------------------------------------------------------
# Game entities (plain dict-based for simple JSON serialization)
# --------------------------------------------------------------------------

class Player:
    def __init__(self, pid: str, nickname: str, ws: WebSocket):
        self.id = pid
        self.nickname = nickname
        self.ws = ws
        self.color = random.choice(PLAYER_COLORS)
        x, y = random_spawn()
        self.x, self.y = x, y
        self.angle = 0.0
        self.max_hp = 100
        self.hp = 100
        self.alive = True
        self.respawn_at = 0.0
        self.weapon = "pistol"
        self.mag = WEAPONS["pistol"]["mag"]
        self.reserve = WEAPONS["pistol"]["reserve_max"]
        self.reloading = False
        self.reload_end = 0.0
        self.last_shot = 0.0
        self.kills = 0
        self.deaths = 0
        self.zombie_kills = 0
        self.trophies = get_trophies(nickname)
        # PvE lives — reset at the start of each match (see start_game),
        # not on every respawn(): losing a life is what a death costs, not
        # something a respawn should undo.
        self.max_lives = LIVES_MAX
        self.lives = LIVES_MAX
        self.eliminated = False  # out of lives — stays a spectator for the rest of this run
        # live input state
        self.keys = {"w": False, "a": False, "s": False, "d": False}
        self.shooting = False
        self.want_reload = False
        # connection lifecycle (supports reload/reconnect without losing the slot)
        self.connected = True
        self.remove_task: Optional[asyncio.Task] = None
        # server-issued secret required to reclaim this slot on reconnect —
        # client_id alone (visible on the wire / guessable) is never enough
        self.secret_token = secrets.token_hex(16)

    def to_state(self):
        return {
            "id": self.id,
            "nickname": self.nickname,
            "color": self.color,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "angle": round(self.angle, 3),
            "hp": max(0, round(self.hp)),
            "max_hp": self.max_hp,
            "alive": self.alive,
            "weapon": self.weapon,
            "weapon_name": WEAPONS[self.weapon]["name"],
            "mag": self.mag,
            "reserve": self.reserve,
            "infinite_reserve": WEAPONS[self.weapon].get("infinite_reserve", False),
            "reloading": self.reloading,
            "kills": self.kills,
            "deaths": self.deaths,
            "lives": self.lives,
            "max_lives": self.max_lives,
            "eliminated": self.eliminated,
        }

    def equip(self, weapon_key):
        self.weapon = weapon_key
        self.mag = WEAPONS[weapon_key]["mag"]
        self.reserve = WEAPONS[weapon_key]["reserve_max"]
        self.reloading = False

    def respawn(self):
        x, y = random_spawn()
        self.x, self.y = x, y
        self.hp = self.max_hp
        self.alive = True
        self.equip("pistol")


class Room:
    def __init__(self, room_id: str):
        self.id = room_id
        self.players: Dict[str, Player] = {}
        self.host_id: Optional[str] = None
        self.mode = "pvp"  # "pvp" | "pve"
        self.status = "lobby"  # "lobby" | "playing"
        self.bullets = []  # dicts: id,x,y,vx,vy,owner,dmg,life
        self.zombies = []  # dicts: id,x,y,hp,max_hp,type,speed,dmg
        self.loot = []  # dicts: id,x,y,type
        self.wave = 0
        self.wave_break_until = 0.0
        self.loop_task: Optional[asyncio.Task] = None
        self.last_loot_spawn = 0.0
        self.match_start = 0.0

    def alive_players(self):
        return [p for p in self.players.values() if p.alive and p.connected]

    async def broadcast(self, message: dict):
        payload = json.dumps(message)
        for pid, p in list(self.players.items()):
            if p.ws is None:
                continue  # slot held for a disconnected player during grace period
            try:
                await p.ws.send_text(payload)
            except Exception:
                mark_disconnected(self, pid)

    def lobby_payload(self):
        return {
            "type": "lobby_update",
            "room_id": self.id,
            "host_id": self.host_id,
            "mode": self.mode,
            "status": self.status,
            "players": [
                {"id": p.id, "nickname": p.nickname, "color": p.color,
                 "is_host": p.id == self.host_id, "trophies": p.trophies}
                for p in self.players.values() if p.connected
            ],
        }

    def pick_new_host(self):
        remaining = [p for p in self.players.values() if p.connected]
        self.host_id = remaining[0].id if remaining else None


ROOMS: Dict[str, Room] = {}


def mark_disconnected(room: Room, pid: str):
    """Flag a player as offline and start the grace-period countdown before
    their slot (and any host rights) are actually released. If they
    reconnect with the same client id before it fires, the task is
    cancelled and nothing is lost."""
    p = room.players.get(pid)
    if p is None or not p.connected:
        return
    p.connected = False
    p.ws = None
    if p.remove_task is None or p.remove_task.done():
        p.remove_task = asyncio.create_task(remove_after_grace(room, pid))


async def remove_after_grace(room: Room, pid: str):
    try:
        await asyncio.sleep(RECONNECT_GRACE)
    except asyncio.CancelledError:
        return
    p = room.players.get(pid)
    if p is None or p.connected:
        return  # reconnected in the meantime
    room.players.pop(pid, None)
    release_session(p.nickname.strip().lower(), room.id, pid)
    if room.host_id == pid:
        room.pick_new_host()
    if not room.players:
        room.status = "lobby"
        ROOMS.pop(room.id, None)
    else:
        await room.broadcast(room.lobby_payload())


# --------------------------------------------------------------------------
# Game loop
# --------------------------------------------------------------------------

async def spawn_wave(room: Room):
    room.wave += 1
    scale = difficulty_scale(room.wave)
    speed_scale = min(1.0 + (scale - 1.0) * 0.35, 1.8)  # speed climbs much slower than hp/dmg and caps out — an unavoidable zombie isn't "harder", it's just unfair

    # Zombie count itself ramps faster once the hard-difficulty point hits,
    # on top of the per-zombie stat scaling below — but capped, since an
    # endless mode means an endlessly growing horde eventually costs more
    # in collision checks (and client-side rendering) than it adds in
    # actual difficulty. Stats keep climbing past the cap via
    # difficulty_scale, so the run still keeps getting harder.
    if room.wave <= HARD_RAMP_WAVE:
        n_zombies = 4 + room.wave * 2
    else:
        extra = room.wave - HARD_RAMP_WAVE
        n_zombies = 4 + HARD_RAMP_WAVE * 2 + extra * 3
    n_zombies = min(n_zombies, MAX_ZOMBIES_PER_WAVE)

    for i in range(n_zombies):
        # New variants unlock progressively so early waves stay readable.
        roll = random.random()
        if room.wave >= 6 and roll < 0.12:
            ztype = "hunter"
        elif room.wave >= 4 and roll < 0.24:
            ztype = "toxic"
        elif room.wave >= 3 and roll < 0.34:
            ztype = "tank"
        elif roll < 0.52:
            ztype = "fast"
        else:
            ztype = "normal"
        stats = ZOMBIE_STATS[ztype]
        x, y = random_edge_point()
        hp = round(stats["hp"] * scale)
        room.zombies.append({
            "id": new_id(),
            "x": x, "y": y,
            "hp": hp, "max_hp": hp,
            "type": ztype,
            "speed": stats["speed"] * speed_scale,
            "dmg": stats["dmg"] * scale,
            "radius": stats["radius"],
            "last_hit": 0.0,
        })

    if room.wave % BOSS_WAVE_INTERVAL == 0:
        tier = room.wave // BOSS_WAVE_INTERVAL  # 1st, 2nd, 3rd... boss — each one tougher
        bstats = ZOMBIE_STATS["boss"]
        x, y = random_edge_point()
        hp = round((bstats["hp"] + (tier - 1) * 350) * scale)
        room.zombies.append({
            "id": new_id(),
            "x": x, "y": y,
            "hp": hp, "max_hp": hp,
            "type": "boss",
            "speed": bstats["speed"] * speed_scale,
            "dmg": (bstats["dmg"] + (tier - 1) * 5) * scale,
            "radius": bstats["radius"],
            "last_hit": 0.0,
        })


def maybe_spawn_loot(room: Room, now: float):
    if len(room.loot) >= MAX_LOOT:
        return
    if now - room.last_loot_spawn < LOOT_SPAWN_INTERVAL:
        return
    room.last_loot_spawn = now
    kind = random.choices(
        ["health", "rifle", "shotgun", "ammo"],
        weights=[40, 22, 18, 20],
    )[0]
    x, y = random_spawn()
    room.loot.append({"id": new_id(), "x": x, "y": y, "type": kind, "spawn_time": now})


def apply_damage_to_player(room: Room, victim: Player, dmg: float, killer: Optional[Player], now: float):
    if not victim.alive:
        return
    victim.hp -= dmg
    if victim.hp <= 0:
        victim.hp = 0
        victim.alive = False
        victim.deaths += 1
        if killer is not None and killer.id != victim.id:
            killer.kills += 1
        if room.mode == "pve":
            victim.lives = max(0, victim.lives - 1)
            if victim.lives > 0:
                victim.respawn_at = now + RESPAWN_TIME
            else:
                victim.eliminated = True  # no lives left — stays down, spectating the rest of the run
        else:
            victim.respawn_at = now + RESPAWN_TIME


async def end_match(room: Room, reason: str):
    """Tally the match, award/deduct trophies, and tell everyone it's over.
    Does not touch room.status/loop_task — the caller (game_loop) is
    responsible for that so there's one place that decides the room is
    no longer 'playing'."""
    results = []

    if room.mode == "pvp":
        ranked = sorted(room.players.values(), key=lambda p: (-p.kills, p.deaths))
        n = len(ranked)
        if n == 2:
            # A straight duel is win/lose, not a podium — keep it simple.
            deltas = [8, -2]
        else:
            placement_reward = [8, 4, 1]  # 1st / 2nd / 3rd place bonus, everyone else loses a couple
            deltas = [placement_reward[i] if i < len(placement_reward) else -2 for i in range(n)]
        for i, p in enumerate(ranked):
            delta = deltas[i]
            p.trophies = add_trophies(p.nickname, delta)
            results.append({
                "nickname": p.nickname, "color": p.color,
                "kills": p.kills, "deaths": p.deaths,
                "trophy_delta": delta, "trophies": p.trophies,
            })
        payload = {
            "type": "game_over", "mode": "pvp", "reason": reason,
            "winner": ranked[0].nickname if ranked else None,
            "results": results,
        }

    else:  # pve
        # Endless mode — no more "victory", just how far the run got. Most
        # of the reward already came in as +1 trophy per cleared wave
        # (paid out live, see the wave-break branch in game_loop); this is
        # just a small flat bonus for finishing the run at all.
        delta = 3
        ranked = sorted(room.players.values(), key=lambda p: -p.zombie_kills)
        for p in ranked:
            p.trophies = add_trophies(p.nickname, delta)
            results.append({
                "nickname": p.nickname, "color": p.color,
                "kills": p.zombie_kills,
                "trophy_delta": delta, "trophies": p.trophies,
            })
        payload = {
            "type": "game_over", "mode": "pve", "reason": reason,
            "wave": room.wave,
            "results": results,
        }

    # One disk write for the whole match, off the event loop thread — not
    # one blocking write per player.
    await save_trophies_async()
    await room.broadcast(payload)


async def game_loop(room: Room):
    room.status = "playing"
    room.match_start = time.time()
    await room.broadcast({"type": "game_start", "mode": room.mode,
                           "map_w": MAP_W, "map_h": MAP_H,
                           "toxic_zone_radius": TOXIC_ZONE_RADIUS})

    if room.mode == "pve":
        await spawn_wave(room)

    try:
        next_tick = time.monotonic()
        while room.status == "playing" and room.players:
            now = time.time()

            # ---- players: movement, shooting, reload, respawn ----
            for p in list(room.players.values()):
                if not p.connected:
                    continue  # frozen in place while their slot is held during reconnect grace
                if not p.alive:
                    if not p.eliminated and now >= p.respawn_at:
                        p.respawn()
                    continue

                mvx = mvy = 0.0
                if p.keys.get("w"):
                    mvy -= 1
                if p.keys.get("s"):
                    mvy += 1
                if p.keys.get("a"):
                    mvx -= 1
                if p.keys.get("d"):
                    mvx += 1
                if mvx or mvy:
                    norm = math.hypot(mvx, mvy)
                    p.x += (mvx / norm) * PLAYER_SPEED * DT
                    p.y += (mvy / norm) * PLAYER_SPEED * DT
                    p.x = clamp(p.x, PLAYER_RADIUS, MAP_W - PLAYER_RADIUS)
                    p.y = clamp(p.y, PLAYER_RADIUS, MAP_H - PLAYER_RADIUS)

                wstats = WEAPONS[p.weapon]
                infinite = wstats.get("infinite_reserve", False)

                if p.want_reload and not p.reloading and p.mag < wstats["mag"] and (infinite or p.reserve > 0):
                    p.reloading = True
                    p.reload_end = now + wstats["reload_time"]
                p.want_reload = False

                if p.reloading and now >= p.reload_end:
                    needed = wstats["mag"] - p.mag
                    take = needed if infinite else min(needed, p.reserve)
                    p.mag += take
                    if not infinite:
                        p.reserve -= take
                    p.reloading = False

                if (p.shooting and not p.reloading and p.mag > 0
                        and now - p.last_shot >= wstats["fire_rate"]):
                    p.last_shot = now
                    p.mag -= 1
                    if p.mag == 0 and (infinite or p.reserve > 0):
                        p.reloading = True
                        p.reload_end = now + wstats["reload_time"]
                    for _ in range(wstats["pellets"]):
                        spread = random.uniform(-wstats["spread"], wstats["spread"])
                        a = p.angle + spread
                        room.bullets.append({
                            "id": new_id(),
                            "x": p.x + math.cos(a) * (PLAYER_RADIUS + 4),
                            "y": p.y + math.sin(a) * (PLAYER_RADIUS + 4),
                            "vx": math.cos(a) * wstats["speed"],
                            "vy": math.sin(a) * wstats["speed"],
                            "owner": p.id,
                            "dmg": wstats["dmg"],
                            "life": 1.1,
                        })

            # ---- bullets: move, expire, collide ----
            zombie_grid = build_zombie_grid(room.zombies) if room.mode == "pve" else None
            surviving_bullets = []
            for b in room.bullets:
                prev_x, prev_y = b["x"], b["y"]
                b["x"] += b["vx"] * DT
                b["y"] += b["vy"] * DT
                b["life"] -= DT
                if (b["life"] <= 0 or b["x"] < 0 or b["x"] > MAP_W
                        or b["y"] < 0 or b["y"] > MAP_H):
                    continue

                hit = False
                owner = room.players.get(b["owner"])

                if room.mode == "pvp":
                    # Among every player the swept segment actually touches,
                    # damage the one whose center is closest to the segment
                    # (roughly: closest along the bullet's path) instead of
                    # whichever happens to come first in dict order — a
                    # bullet shouldn't be able to skip past a nearer target
                    # to hit one standing behind it.
                    best, best_d = None, None
                    for p in room.players.values():
                        if p.id == b["owner"] or not p.alive or not p.connected:
                            continue
                        d = dist_point_to_segment(p.x, p.y, prev_x, prev_y, b["x"], b["y"])
                        if d <= PLAYER_RADIUS + BULLET_RADIUS and (best is None or d < best_d):
                            best, best_d = p, d
                    if best is not None:
                        apply_damage_to_player(room, best, b["dmg"], owner, now)
                        hit = True
                else:  # pve
                    best, best_d = None, None
                    for zi in nearby_zombie_indices(zombie_grid, b["x"], b["y"]):
                        z = room.zombies[zi]
                        if z["hp"] <= 0:
                            continue  # already dead this tick, just hasn't been swept out yet
                        d = dist_point_to_segment(z["x"], z["y"], prev_x, prev_y, b["x"], b["y"])
                        if d <= z["radius"] + BULLET_RADIUS and (best is None or d < best_d):
                            best, best_d = z, d
                    if best is not None:
                        best["hp"] -= b["dmg"]
                        hit = True
                        if best["hp"] <= 0 and owner is not None:
                            owner.zombie_kills += 1

                if not hit:
                    surviving_bullets.append(b)
            room.bullets = surviving_bullets
            if room.mode == "pve":
                # Sweep out anything that died this tick once, instead of
                # rebuilding the list after every single bullet hit.
                room.zombies = [z for z in room.zombies if z["hp"] > 0]

            # ---- PvE zombie AI ----
            if room.mode == "pve":
                if room.players and all(p.eliminated for p in room.players.values()):
                    # Nobody has a life left — this is the only way an
                    # endless run actually ends (short of everyone leaving).
                    await end_match(room, "wiped")
                    break

                alive_players = room.alive_players()
                for z in room.zombies:
                    if alive_players:
                        target = min(alive_players, key=lambda p: dist(z["x"], z["y"], p.x, p.y))
                        d = dist(z["x"], z["y"], target.x, target.y)
                        if d > 1:
                            dx = (target.x - z["x"]) / d
                            dy = (target.y - z["y"]) / d
                            z["x"] = clamp(z["x"] + dx * z["speed"] * DT, 0, MAP_W)
                            z["y"] = clamp(z["y"] + dy * z["speed"] * DT, 0, MAP_H)
                        if d <= z["radius"] + PLAYER_RADIUS and now - z["last_hit"] > 0.7:
                            z["last_hit"] = now
                            apply_damage_to_player(room, target, z["dmg"], None, now)
                    else:
                        # Nobody alive to chase (everyone mid-respawn or the
                        # room is empty of live targets) — wander instead of
                        # freezing in place, so it doesn't look broken.
                        if random.random() < 0.02 or "wander_angle" not in z:
                            z["wander_angle"] = random.uniform(0, 2 * math.pi)
                        wa = z["wander_angle"]
                        z["x"] = clamp(z["x"] + math.cos(wa) * z["speed"] * 0.35 * DT, 0, MAP_W)
                        z["y"] = clamp(z["y"] + math.sin(wa) * z["speed"] * 0.35 * DT, 0, MAP_H)

                    # Toxic zombies also hurt anyone standing in their aura,
                    # continuously and independent of the melee-contact
                    # check above — it's a lingering damage zone, not a hit.
                    if z["type"] == "toxic":
                        for p in room.players.values():
                            if not p.alive or not p.connected:
                                continue
                            if dist(p.x, p.y, z["x"], z["y"]) <= TOXIC_ZONE_RADIUS:
                                apply_damage_to_player(room, p, TOXIC_ZONE_DPS * DT, None, now)

                if not room.zombies:
                    if room.wave_break_until == 0.0:
                        room.wave_break_until = now + WAVE_BREAK_TIME
                        # Wave cleared — pay out +1 trophy per player right
                        # now rather than waiting for the run to end, since
                        # an endless mode has no natural "end of match"
                        # point where a lump sum would make sense.
                        for p in room.players.values():
                            p.trophies = add_trophies(p.nickname, 1)
                        await save_trophies_async()
                        await room.broadcast({"type": "wave_cleared", "wave": room.wave})
                    elif now >= room.wave_break_until:
                        room.wave_break_until = 0.0
                        await spawn_wave(room)

            elif room.mode == "pvp":
                time_up = (now - room.match_start) >= PVP_TIME_LIMIT
                kills_hit = any(p.kills >= PVP_KILL_LIMIT for p in room.players.values())
                if time_up or kills_hit:
                    await end_match(room, "kill_limit" if kills_hit else "time_limit")
                    break

            # ---- loot ----
            maybe_spawn_loot(room, now)
            remaining_loot = []
            for item in room.loot:
                if now - item.get("spawn_time", now) >= LOOT_LIFETIME:
                    continue  # despawned — frees a slot for maybe_spawn_loot next tick
                picker, picker_d = None, None
                for p in room.players.values():
                    if not p.alive or not p.connected:
                        continue
                    d = dist(p.x, p.y, item["x"], item["y"])
                    if d <= PLAYER_RADIUS + 14 and (picker is None or d < picker_d):
                        picker, picker_d = p, d
                if picker is not None:
                    if item["type"] == "health":
                        picker.hp = min(picker.max_hp, picker.hp + 45)
                    elif item["type"] == "ammo":
                        picker.reserve = min(WEAPONS[picker.weapon]["reserve_max"], picker.reserve + 30)
                    else:
                        picker.equip(item["type"])
                else:
                    remaining_loot.append(item)
            room.loot = remaining_loot

            # ---- broadcast state ----
            leaderboard = sorted(
                (p for p in room.players.values() if p.connected),
                key=lambda p: (-p.kills, p.deaths),
            )
            state = {
                "type": "state",
                "mode": room.mode,
                "map_w": MAP_W, "map_h": MAP_H,
                "players": [p.to_state() for p in room.players.values() if p.connected],
                "bullets": [
                    {"id": b["id"], "x": round(b["x"], 1), "y": round(b["y"], 1),
                     "vx": round(b["vx"], 1), "vy": round(b["vy"], 1)}
                    for b in room.bullets
                ],
                "zombies": [
                    {"id": z["id"], "x": round(z["x"], 1), "y": round(z["y"], 1),
                     "hp": z["hp"], "max_hp": z["max_hp"], "type": z["type"]}
                    for z in room.zombies
                ],
                "loot": room.loot,
                "wave": room.wave,
                "zombies_left": len(room.zombies),
                "leaderboard": [
                    {"nickname": p.nickname, "kills": p.kills, "deaths": p.deaths, "color": p.color}
                    for p in leaderboard
                ],
            }
            await room.broadcast(state)

            # Sleep until the next scheduled tick boundary rather than a
            # flat DT — a flat sleep-after-work drifts the effective tick
            # rate below TICK_RATE as soon as per-tick processing (many
            # bullets/zombies) takes a non-trivial slice of the budget.
            next_tick += DT
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            else:
                next_tick = time.monotonic()  # fell behind — don't try to burst-catch-up
    finally:
        room.status = "lobby"
        room.loop_task = None
        room.wave_break_until = 0.0
        await room.broadcast(room.lobby_payload())


# --------------------------------------------------------------------------
# WebSocket endpoint
# --------------------------------------------------------------------------

@app.websocket("/ws/{room_id}/{nickname}")
async def ws_endpoint(websocket: WebSocket, room_id: str, nickname: str,
                       client_id: str = ""):
    await websocket.accept()

    room_id = room_id.strip()[:24] or "default"
    nickname = nickname.strip()[:16] or "Player"
    pid = (client_id.strip() or new_id())[:40]

    # The reconnect token and account password are secrets — unlike
    # room_id/nickname/client_id (identifiers, not secrets, and client_id
    # alone was never trusted for anything, see the module docstring),
    # these never go in the URL. A query string routinely ends up in
    # reverse-proxy / access logs even behind wss://, so the client sends
    # them in the first WebSocket message instead, right after the
    # handshake completes.
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=8.0)
        auth_msg = json.loads(raw)
    except (asyncio.TimeoutError, json.JSONDecodeError, WebSocketDisconnect):
        await websocket.close()
        return
    if not isinstance(auth_msg, dict) or auth_msg.get("type") != "auth":
        await websocket.close()
        return
    token = str(auth_msg.get("token") or "")
    password = str(auth_msg.get("password") or "")

    room = ROOMS.setdefault(room_id, Room(room_id))

    existing = room.players.get(pid)
    if existing is not None and token and existing.secret_token == token:
        # Verified reconnect — this is the same slot the server itself
        # handed a secret token out for. Reuse the slot, stats, and host
        # status. Already proven who they are, so no password check here —
        # but that also means we must NOT let this request rename the
        # slot: the token only proves ownership of THIS slot, not of
        # whatever nickname the client happens to send this time. Silently
        # keep the nickname exactly as registered; otherwise anyone
        # holding a valid token for their own slot could relabel it with
        # someone else's nickname on reconnect and inherit that other
        # account's trophies/session lock without ever entering its
        # password.
        if existing.remove_task is not None:
            existing.remove_task.cancel()
            existing.remove_task = None
        existing.ws = websocket
        existing.connected = True
        player = existing
        # Re-assert the claim in case a stale entry was ever cleaned up
        # from under this exact slot — this is still the one legitimate
        # owner of it.
        claim_session(player.nickname.strip().lower(), room.id, pid)
    else:
        # Minting a brand-new identity — this is the point where the
        # password-protected account needs to be checked, since nothing
        # else here proves who's actually typing it.
        ok, err = await claim_or_verify_account(nickname, password)
        if not ok:
            await websocket.send_text(json.dumps({"type": "error", "message": err}))
            await websocket.close()
            return

        # One account, one active slot at a time — anywhere on the server,
        # not just this room. Stops the same login being played from two
        # tabs/devices simultaneously (and, as a side effect, stops two
        # simultaneous sessions from each collecting trophies for the same
        # account at match end).
        key = nickname.strip().lower()
        if session_conflict(key):
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "This account is already playing on another device or tab. "
                           "Finish that session first (leave the lobby or close the tab), then try again.",
            }))
            await websocket.close()
            return

        pid = new_id()
        player = Player(pid, nickname, websocket)
        room.players[pid] = player
        if room.host_id is None:
            room.host_id = pid
        claim_session(key, room.id, pid)

    await websocket.send_text(json.dumps({
        "type": "joined", "player_id": pid, "token": player.secret_token,
    }))
    await room.broadcast(room.lobby_payload())
    if room.status == "playing":
        await websocket.send_text(json.dumps({"type": "game_start", "mode": room.mode,
                                                "map_w": MAP_W, "map_h": MAP_H,
                                                "toxic_zone_radius": TOXIC_ZONE_RADIUS}))

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            mtype = msg.get("type")

            if mtype == "set_mode" and pid == room.host_id and room.status == "lobby":
                if msg.get("mode") in ("pvp", "pve"):
                    room.mode = msg["mode"]
                    await room.broadcast(room.lobby_payload())

            elif mtype == "start_game" and pid == room.host_id and room.status == "lobby":
                connected_count = sum(1 for p in room.players.values() if p.connected)
                if room.mode == "pvp" and connected_count < 2:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "PvP requires at least 2 players.",
                    }))
                elif room.loop_task is None:
                    for p in room.players.values():
                        p.respawn()
                        p.kills = 0
                        p.deaths = 0
                        p.zombie_kills = 0
                        p.lives = p.max_lives
                        p.eliminated = False
                    room.bullets, room.zombies, room.loot, room.wave = [], [], [], 0
                    room.loop_task = asyncio.create_task(game_loop(room))

            elif mtype == "input":
                keys = msg.get("keys") or {}
                player.keys = {
                    "w": bool(keys.get("w")),
                    "a": bool(keys.get("a")),
                    "s": bool(keys.get("s")),
                    "d": bool(keys.get("d")),
                }
                if "angle" in msg:
                    player.angle = float(msg["angle"])
                player.shooting = bool(msg.get("shooting"))
                if msg.get("reload"):
                    player.want_reload = True

            elif mtype == "return_to_lobby" and pid == room.host_id:
                room.status = "lobby"
                await room.broadcast(room.lobby_payload())

            elif mtype == "leave" and room.status == "lobby":
                # Explicit "leave lobby" — remove the slot immediately
                # instead of waiting out the reconnect grace period, so the
                # room and host status update for everyone right away.
                room.players.pop(pid, None)
                release_session(player.nickname.strip().lower(), room.id, pid)
                if room.host_id == pid:
                    room.pick_new_host()
                if not room.players:
                    room.status = "lobby"
                    ROOMS.pop(room_id, None)
                else:
                    await room.broadcast(room.lobby_payload())
                await websocket.close()
                return

    except WebSocketDisconnect:
        pass
    finally:
        current = room.players.get(pid)
        # Only tear this connection down if it's still the active one for
        # this slot — a reconnect may already have replaced it by the time
        # this finally block runs.
        if current is not None and current.ws is websocket:
            mark_disconnected(room, pid)
            await room.broadcast(room.lobby_payload())


if __name__ == "__main__":
    import uvicorn
    # $PORT is how Render (and most PaaS hosts) tell your app which port
    # to bind — locally it's just not set, so this still works with
    # `python main.py` on your machine, defaulting to 8000.
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
