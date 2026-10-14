import { Translation } from "./definition"

export function arabicNotes(count: number) {
  return count === 0 ? "لا توجد ملاحظات" : count === 1 ? "ملاحظة واحدة" : count === 2 ? "ملاحظتان" : `${count} ${count % 100 >= 3 && count % 100 <= 10 ? "ملاحظات" : "ملاحظة"}`
}

export default {
  propertyDefaults: {
    title: "بلا عنوان",
    description: "ملاحظات عبدالكريم الخطيب",
  },
  direction: "rtl" as const,
  components: {
    callout: {
      note: "ملاحظة",
      abstract: "ملخص",
      info: "معلومات",
      todo: "مهمة",
      tip: "نصيحة",
      success: "نجاح",
      question: "سؤال",
      warning: "تحذير",
      failure: "فشل",
      danger: "خطر",
      bug: "خلل",
      example: "مثال",
      quote: "اقتباس",
    },
    backlinks: {
      title: "روابط واردة",
      noBacklinksFound: "لا توجد ملاحظات تشير إلى هذه الصفحة",
    },
    themeToggle: {
      lightMode: "الوضع النهاري",
      darkMode: "الوضع الليلي",
    },
    explorer: {
      title: "تصفح الملاحظات",
    },
    readerMode: {
      title: "وضع القراءة",
    },
    footer: {
      createdWith: "أُنشئ باستخدام",
    },
    graph: {
      title: "خريطة الروابط",
    },
    recentNotes: {
      title: "آخر الملاحظات",
      seeRemainingMore: ({ remaining }) => `تصفح المزيد (${remaining}) ←`,
    },
    transcludes: {
      transcludeOf: ({ targetSlug }) => `مقتبس من ${targetSlug}`,
      linkToOriginal: "عرض الملاحظة الأصلية",
    },
    search: {
      title: "بحث",
      searchBarPlaceholder: "ابحث في الملاحظات…",
    },
    tableOfContents: {
      title: "في هذه الصفحة",
    },
    contentMeta: {
      readingTime: ({ minutes }) =>
        minutes <= 1
          ? `دقيقة للقراءة`
          : minutes == 2
            ? `دقيقتان للقراءة`
            : `${minutes} ${minutes % 100 >= 3 && minutes % 100 <= 10 ? "دقائق" : "دقيقة"} للقراءة`,
    },
  },
  pages: {
    rss: {
      recentNotes: "آخر الملاحظات",
      lastFewNotes: ({ count }) => `آخر ${count} ملاحظة`,
    },
    error: {
      title: "الصفحة غير موجودة",
      notFound: "تعذر العثور على هذه الصفحة. ربما نُقلت أو لم تُنشر بعد.",
      home: "العودة إلى الرئيسية",
    },
    folderContent: {
      folder: "مجلد",
      itemsUnderFolder: ({ count }) => `${arabicNotes(count)} في هذا المجلد`,
    },
    tagContent: {
      tag: "الوسم",
      tagIndex: "جميع الوسوم",
      itemsUnderTag: ({ count }) => `${arabicNotes(count)} بهذا الوسم`,
      showingFirst: ({ count }) => `عرض أول ${count} من الوسوم`,
      totalTags: ({ count }) => count === 0 ? "لا توجد وسوم" : count === 1 ? "وسم واحد" : count === 2 ? "وسمان" : `${count} ${count % 100 >= 3 && count % 100 <= 10 ? "وسوم" : "وسمًا"}`,
    },
  },
} as const satisfies Translation
