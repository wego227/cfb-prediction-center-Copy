"""Conservative mapping of sportsbook display names to CFBD canonical names."""
import re
import unicodedata
from difflib import get_close_matches

ALIASES = {
    "miami hurricanes": "Miami",
    "miami fl": "Miami",
    "miami florida": "Miami",
    "miami ohio redhawks": "Miami (OH)",
    "miami oh": "Miami (OH)",
    "ole miss rebels": "Ole Miss",
    "mississippi rebels": "Ole Miss",
    "usc trojans": "USC",
    "southern california trojans": "USC",
    "uconn huskies": "UConn",
    "connecticut huskies": "UConn",
    "ucf knights": "UCF",
    "central florida knights": "UCF",
    "smu mustangs": "SMU",
    "southern methodist mustangs": "SMU",
    "byu cougars": "BYU",
    "brigham young cougars": "BYU",
    "tcu horned frogs": "TCU",
    "texas christian horned frogs": "TCU",
    "utsa roadrunners": "UTSA",
    "utep miners": "UTEP",
    "uab blazers": "UAB",
    "lsu tigers": "LSU",
    "pitt panthers": "Pittsburgh",
    "pittsburgh panthers": "Pittsburgh",
    "nc state wolfpack": "NC State",
    "north carolina state wolfpack": "NC State",
    "app state mountaineers": "App State",
    "appalachian state mountaineers": "App State",
    "jmu dukes": "James Madison",
    "umass minutemen": "Massachusetts",
    "massachusetts minutemen": "Massachusetts",
    "fiu panthers": "Florida International",
    "florida international panthers": "Florida International",
    "fau owls": "Florida Atlantic",
    "florida atlantic owls": "Florida Atlantic",
    "usf bulls": "South Florida",
    "south florida bulls": "South Florida",
    "unlv rebels": "UNLV",
    "hawaii rainbow warriors": "Hawai'i",
    "hawaii warriors": "Hawai'i",
    "san jose state spartans": "San José State",
    "san jose st spartans": "San José State",
    "ul monroe warhawks": "UL Monroe",
    "louisiana monroe warhawks": "UL Monroe",
    "louisiana ragin cajuns": "Louisiana",
    "louisiana lafayette ragin cajuns": "Louisiana",
    "western kentucky hilltoppers": "Western Kentucky",
    "middle tennessee blue raiders": "Middle Tennessee",
    "southern miss golden eagles": "Southern Mississippi",
    "southern mississippi golden eagles": "Southern Mississippi",
    "southern miss golden eagles": "Southern Mississippi",
    "army black knights": "Army",
    "navy midshipmen": "Navy",
    "air force falcons": "Air Force",
    "cal golden bears": "California",
    "california golden bears": "California",
    "colorado state rams": "Colorado State",
    "boise state broncos": "Boise State",
    "kent state golden flashes": "Kent State",
    "ohio bobcats": "Ohio",
    "ohio state buckeyes": "Ohio State",
    "washington state cougars": "Washington State",
    "oregon state beavers": "Oregon State",
    "georgia state panthers": "Georgia State",
    "georgia southern eagles": "Georgia Southern",
    "texas state bobcats": "Texas State",
    "new mexico state aggies": "New Mexico State",
    "new mexico lobos": "New Mexico",
    "north texas mean green": "North Texas",
    "east carolina pirates": "East Carolina",
    "coastal carolina chanticleers": "Coastal Carolina",
    "charlotte 49ers": "Charlotte",
    "sam houston bearkats": "Sam Houston",
    "sam houston state bearkats": "Sam Houston",
    "north dakota state bison": "North Dakota State",
    "appalachian state mountaineers": "App State",
    "kennesaw state owls": "Kennesaw State",
    "delaware blue hens": "Delaware",
    "missouri state bears": "Missouri State",
}

def norm(s):
    value=unicodedata.normalize("NFKD",str(s))
    value="".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9]+"," ",value.lower())).strip()

def match_team(display, canonical):
    """Return (canonical name or None, method, suggestions). Never guess ambiguous matches."""
    names=sorted({str(x) for x in canonical if str(x).strip()})
    norm_to_names={}
    for name in names:
        norm_to_names.setdefault(norm(name),[]).append(name)
    key=norm(display)
    if key in norm_to_names and len(norm_to_names[key])==1:
        return norm_to_names[key][0],"exact",[]
    alias=ALIASES.get(key)
    if alias:
        found=norm_to_names.get(norm(alias),[])
        if len(found)==1:
            return found[0],"alias",[]
    # Avoid prefix collisions: Southern Mississippi must never map to "Southern".
    # If the intended FBS name is missing from CFBD history, block prediction.
    if key.startswith("southern mississippi ") or key.startswith("southern miss "):
        return None,"unmatched",["Southern Mississippi (missing from CFBD history)"]
    # Odds feed typically appends a mascot: "Iowa State Cyclones" -> "Iowa State".
    # Require unique longest canonical prefix; don't silently conflate Miami/Miami (OH).
    prefixes=[n for n in names if key.startswith(norm(n)+" ")]
    if prefixes:
        longest=max(len(norm(n)) for n in prefixes)
        best=[n for n in prefixes if len(norm(n))==longest]
        if len(best)==1:
            return best[0],"unique_prefix",[]
    # Suggestions are diagnostic only, NEVER used automatically.
    suggestions=get_close_matches(key,list(norm_to_names),n=3,cutoff=.48)
    return None,"unmatched",[norm_to_names[x][0] for x in suggestions]
