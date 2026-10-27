"""v4 최종 보고서 영문판. 국문판과 같은 결과·그림을 쓰고, 본문은 report_v4_body_en.html,
생성된 표·그림의 한국어는 i18n_en 사전으로 바꾼다. 페이지에 한글이 남으면 오류로 알린다.

사용법: python make_report_v4_en.py <v4 엑셀> <results_v4> <v3 엑셀> <results_v3> <출력 html>
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i18n_en  # noqa: E402
import make_report_v4  # noqa: E402

if __name__ == "__main__":
    out = sys.argv[5]
    make_report_v4.main(*sys.argv[1:6], body_file="report_v4_body_en.html",
                        title="Pepper Anthracnose Analysis", tr=i18n_en.tr)
    page = open(out).read()
    page = re.sub(r"/\*[^*]*[\uac00-\ud7a3][^*]*\*/", "", page)   # 한국어 CSS 주석 제거
    open(out, "w").write(page)
    left = i18n_en.leftover(page)
    if left:
        raise SystemExit(f"untranslated Korean left: {left[:20]}")
    print("no Korean left")
