from src.contour_match import match_results

if __name__ == '__main__':
    excel_path= "../data/2n.xlsx"
    results = match_results(excel_path)
    print(results)