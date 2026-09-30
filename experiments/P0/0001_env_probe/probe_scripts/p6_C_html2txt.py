"""usage: p6_C_html2txt.py <url> [maxchars] -- fetch a web page, strip tags (comments kept marked), print text. evidence helper for RUN-0001 §6 group C"""
import sys, re, html, urllib.request
url = sys.argv[1]; mx = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
t = urllib.request.urlopen(req, timeout=60).read().decode('utf-8', 'ignore')
t = re.sub(r'<script.*?</script>', '', t, flags=re.S); t = re.sub(r'<style.*?</style>', '', t, flags=re.S)
t = re.sub(r'<!--(.*?)-->', lambda m: ' [HTML-COMMENT: ' + re.sub(r'<[^>]+>', ' ', m.group(1)) + ' ]', t, flags=re.S)
links = re.findall(r'href="([^"]+)"', t)
txt = re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', t)))
print(txt[:mx]); print('LINKS:', links[:200])
