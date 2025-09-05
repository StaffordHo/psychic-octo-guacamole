#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include <ZXing/ReadBarcode.h>
#include <ZXing/ImageView.h>
#include <ZXing/BarcodeFormat.h>
#include <ZXing/ReaderOptions.h>

namespace fs = std::filesystem;

// -------------------- CLI parsing --------------------
struct Args {
    fs::path path;
    fs::path debugDir;
    std::vector<cv::Rect> rois; // user ROIs
};

static bool parseArgs(int argc, char* argv[], Args& a)
{
    if (argc < 2) return false;
    a.path = fs::path(argv[1]);
    for (int i = 2; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--save-debug" && i + 1 < argc) {
            a.debugDir = fs::path(argv[++i]);
        } else if (s == "--roi" && i + 4 < argc) {
            int x = std::stoi(argv[++i]);
            int y = std::stoi(argv[++i]);
            int w = std::stoi(argv[++i]);
            int h = std::stoi(argv[++i]);
            a.rois.emplace_back(x, y, w, h);
        } else {
            std::cerr << "Unknown/invalid arg: " << s << "\n";
            return false;
        }
    }
    return true;
}

// -------------------- Utils --------------------
static inline cv::Mat ensureGray8(const cv::Mat& in)
{
    cv::Mat g;
    if (in.empty()) return g;
    if (in.channels() == 3) cv::cvtColor(in, g, cv::COLOR_BGR2GRAY);
    else if (in.channels() == 4) cv::cvtColor(in, g, cv::COLOR_BGRA2GRAY);
    else g = in.clone();
    if (g.type() != CV_8UC1) g.convertTo(g, CV_8U);
    return g;
}

static void saveDebug(const fs::path& dir, const std::string& name, const cv::Mat& img)
{
    if (dir.empty()) return;
    fs::create_directories(dir);
    cv::imwrite((dir / name).string(), img);
}

static ZXing::Result decodeZX(const cv::Mat& imgGray)
{
    ZXing::ImageView view(imgGray.data, imgGray.cols, imgGray.rows,
                          ZXing::ImageFormat::Lum, static_cast<int>(imgGray.step));
    ZXing::ReaderOptions opts;
    opts.setFormats(ZXing::BarcodeFormat::DataMatrix);
    opts.setTryHarder(true);
    opts.setTryRotate(true);
    return ZXing::ReadBarcode(view, opts);
}

static ZXing::Result tryRotations(const cv::Mat& g)
{
    std::vector<int> angles = {0,90,180,270};
    for (int ang : angles) {
        cv::Mat r;
        if (ang == 0) r = g;
        else if (ang == 90) cv::rotate(g, r, cv::ROTATE_90_CLOCKWISE);
        else if (ang == 180) cv::rotate(g, r, cv::ROTATE_180);
        else cv::rotate(g, r, cv::ROTATE_90_COUNTERCLOCKWISE);
        ZXing::Result z = decodeZX(r);
        if (z.isValid()) return z;
    }
    return ZXing::Result();
}

// Build a bunch of variants
static void buildVariants(const cv::Mat& gray, std::vector<std::pair<std::string, cv::Mat>>& out)
{
    out.clear();
    out.emplace_back("raw", gray);

    // multi-scale
    std::vector<double> scales = {1.5, 2.0, 3.0};
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8,8));

    for (double sc : scales) {
        cv::Mat up; cv::resize(gray, up, cv::Size(), sc, sc, cv::INTER_CUBIC);
        out.emplace_back("up" + std::to_string(int(sc*10)), up);

        cv::Mat upC; clahe->apply(up, upC);
        out.emplace_back("upC" + std::to_string(int(sc*10)), upC);

        cv::Mat med; cv::medianBlur(upC, med, 3);
        // Otsu both polarities
        cv::Mat otsu, otsuInv;
        cv::threshold(med, otsu, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        cv::threshold(med, otsuInv, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
        out.emplace_back("otsu" + std::to_string(int(sc*10)), otsu);
        out.emplace_back("otsuInv" + std::to_string(int(sc*10)), otsuInv);

        // Adaptive
        cv::Mat adM, adMI, adG, adGI;
        cv::adaptiveThreshold(med, adM,  255, cv::ADAPTIVE_THRESH_MEAN_C,     cv::THRESH_BINARY,     15, 10);
        cv::adaptiveThreshold(med, adMI, 255, cv::ADAPTIVE_THRESH_MEAN_C,     cv::THRESH_BINARY_INV, 15, 10);
        cv::adaptiveThreshold(med, adG,  255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY,     15, 10);
        cv::adaptiveThreshold(med, adGI, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 15, 10);
        out.emplace_back("adM" + std::to_string(int(sc*10)), adM);
        out.emplace_back("adMI" + std::to_string(int(sc*10)), adMI);
        out.emplace_back("adG" + std::to_string(int(sc*10)), adG);
        out.emplace_back("adGI" + std::to_string(int(sc*10)), adGI);

        // Morph close on inverse variants (often helps dot-peen)
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {3,3});
        cv::Mat c1, c2, c3;
        cv::morphologyEx(otsuInv, c1, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(adMI,   c2, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(adGI,   c3, cv::MORPH_CLOSE, kernel);
        out.emplace_back("closeOtsuInv" + std::to_string(int(sc*10)), c1);
        out.emplace_back("closeAdMI"    + std::to_string(int(sc*10)), c2);
        out.emplace_back("closeAdGI"    + std::to_string(int(sc*10)), c3);

        // Black-hat (emphasize dark dots on bright)
        cv::Mat bh;
        cv::morphologyEx(med, bh, cv::MORPH_BLACKHAT, cv::getStructuringElement(cv::MORPH_RECT, {5,5}));
        out.emplace_back("blackhat" + std::to_string(int(sc*10)), bh);
    }
}

// Optional: search for square-ish ROIs on a binary image
static std::vector<cv::Rect> findSquareishROIs(const cv::Mat& binary)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> rois;
    double imgArea = static_cast<double>(binary.rows * binary.cols);

    for (const auto& c : contours) {
        if (c.size() < 20) continue;
        cv::RotatedRect rr = cv::minAreaRect(c);
        float w = rr.size.width, h = rr.size.height;
        if (w < 6 || h < 6) continue;

        double ar = (w > h) ? (w / h) : (h / w);
        if (ar > 1.5) continue;

        double area = w * h;
        if (area < 0.005 * imgArea || area > 0.7 * imgArea) continue;

        cv::Rect br = cv::boundingRect(c);
        int pad = std::max(4, std::min(br.width, br.height) / 15);
        cv::Rect p = br;
        p.x = std::max(0, p.x - pad);
        p.y = std::max(0, p.y - pad);
        p.width  = std::min(binary.cols - p.x, p.width  + 2*pad);
        p.height = std::min(binary.rows - p.y, p.height + 2*pad);
        rois.push_back(p);
    }

    std::sort(rois.begin(), rois.end(), [](const cv::Rect& a, const cv::Rect& b){ return a.area() > b.area(); });
    if (rois.size() > 12) rois.resize(12);
    return rois;
}

static bool decodeImageOnce(const cv::Mat& img, const fs::path& dbgDir, const std::string& tag, std::string& outText)
{
    cv::Mat gray = ensureGray8(img);

    // Try raw
    ZXing::Result r = tryRotations(gray);
    if (r.isValid()) { outText = r.text(); return true; }

    // Variants
    std::vector<std::pair<std::string, cv::Mat>> vars;
    buildVariants(gray, vars);

    for (auto& v : vars) {
        if (!dbgDir.empty()) saveDebug(dbgDir, tag + "_" + v.first + ".png", v.second);
        ZXing::Result rr = tryRotations(v.second);
        if (rr.isValid()) { outText = rr.text(); return true; }
    }

    // ROI hunt on binary-looking variants
    for (auto& v : vars) {
        // Threshold to binary for contouring if needed
        cv::Mat bin;
        if (v.second.channels() == 1) {
            cv::threshold(v.second, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        } else {
            continue;
        }
        auto rois = findSquareishROIs(bin);
        for (size_t i = 0; i < rois.size(); ++i) {
            cv::Rect rbox = rois[i] & cv::Rect(0,0, v.second.cols, v.second.rows);
            if (rbox.width <= 0 || rbox.height <= 0) continue;
            cv::Mat roi = v.second(rbox).clone();

            // upscale ROI again
            cv::Mat roiUp; cv::resize(roi, roiUp, cv::Size(), 1.8, 1.8, cv::INTER_CUBIC);
            if (!dbgDir.empty()) saveDebug(dbgDir, tag + "_roi_" + v.first + "_" + std::to_string(i) + ".png", roiUp);

            ZXing::Result rr = tryRotations(roiUp);
            if (rr.isValid()) { outText = rr.text(); return true; }
        }
    }

    return false;
}

static void processPath(const fs::path& p, const Args& a)
{
    if (fs::is_directory(p)) {
        for (const auto& e : fs::directory_iterator(p)) {
            auto ext = e.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tif" || ext == ".tiff") {
                processPath(e.path(), a);
            }
        }
        return;
    }

    std::cout << "Processing: " << p.filename().string() << "\n";
    cv::Mat img = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
    if (img.empty()) { std::cout << "  Failed to load image.\n\n"; return; }

    // If user provided ROIs, try them first (each ROI will also get all variants)
    for (size_t i = 0; i < a.rois.size(); ++i) {
        cv::Rect r = a.rois[i] & cv::Rect(0,0, img.cols, img.rows);
        if (r.width <= 0 || r.height <= 0) continue;
        cv::Mat roi = img(r).clone();
        std::string text;
        if (decodeImageOnce(roi, a.debugDir, "userROI" + std::to_string(i), text)) {
            std::cout << "  Decoded (ROI#" << i << "): " << text << "\n\n";
            return;
        }
    }

    // Otherwise full-frame search
    std::string text;
    if (decodeImageOnce(img, a.debugDir, "full", text)) {
        std::cout << "  Decoded: " << text << "\n\n";
    } else {
        std::cout << "  Not detected\n\n";
    }
}

int main(int argc, char* argv[])
{
    Args args;
    if (!parseArgs(argc, argv, args) || args.path.empty()) {
        std::cout <<
            "Usage:\n"
            "  " << argv[0] << " <image_or_folder> [--save-debug <dir>] [--roi x y w h] [--roi x y w h]\n";
        return 1;
    }

    if (!fs::exists(args.path)) {
        std::cerr << "Path does not exist: " << args.path.string() << "\n";
        return 1;
    }

    processPath(args.path, args);
    return 0;
}
