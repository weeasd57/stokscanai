interface IndexData {
  date: string;
  symbol: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  adjusted_close: number;
}

export function calculateIndexWinRate(startDate: string, endDate: string): number {
  try {
    // في بيئة التطوير، سنقوم بقراءة الملف مباشرة
    // في بيئة الإنتاج، يجب أن تكون هذه البيانات متاحة من API

    // محاكاة بيانات EGX30 - يجب استبدالها بالبيانات الحقيقية
    const mockIndexData: IndexData[] = [
      { date: "2024-01-01", symbol: "EGX:EGX30", open: 15000, high: 15200, low: 14800, close: 15100, volume: 0, adjusted_close: 15100 },
      { date: "2024-01-31", symbol: "EGX:EGX30", open: 15100, high: 15300, low: 14900, close: 15250, volume: 0, adjusted_close: 15250 },
      { date: "2024-02-01", symbol: "EGX:EGX30", open: 15250, high: 15400, low: 15100, close: 15350, volume: 0, adjusted_close: 15350 },
      { date: "2024-02-29", symbol: "EGX:EGX30", open: 15350, high: 15500, low: 15200, close: 15480, volume: 0, adjusted_close: 15480 },
    ];

    // تحويل التواريخ
    const start = new Date(startDate);
    const end = new Date(endDate);

    // تصفية البيانات للفترة المطلوبة
    const periodData = mockIndexData.filter(item => {
      const itemDate = new Date(item.date);
      return itemDate >= start && itemDate <= end;
    });

    if (periodData.length < 2) {
      return 0.0;
    }

    // حساب نسبة التغيير
    const startPrice = periodData[0].close;
    const endPrice = periodData[periodData.length - 1].close;

    const winRate = ((endPrice - startPrice) / startPrice) * 100;
    return Math.round(winRate * 100) / 100; // تقريب لمنزلتين عشريتين

  } catch (error) {
    console.error('Error calculating index win rate:', error);
    return 0.0;
  }
}

export async function getIndexWinRateFromAPI(startDate: string, endDate: string): Promise<number> {
  try {
    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "";
    const response = await fetch(`${baseUrl}/index-winrate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        start_date: startDate,
        end_date: endDate,
        index: 'EGX30'
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data.win_rate || 0.0;

  } catch (error) {
    console.error('Error fetching index win rate from API:', error);
    // الرجوع إلى الحساب المحلي في حالة فشل API
    return calculateIndexWinRate(startDate, endDate);
  }
}

// دالة مساعدة لتحسين أداء الحساب (تم تعطيل الكاش المحلي بناءً على طلب المستخدم)
export function getCachedIndexWinRate(startDate: string, endDate: string): number {
  return calculateIndexWinRate(startDate, endDate);
}
